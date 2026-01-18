import argparse
import collections
import datetime
import glob
import logging
import os
import pathlib
import sys
import time
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import Audio
import fluidsynth

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameters and Constants (fully parameterized with defaults)
CONFIG = {
    'SAMPLING_RATE': 16000, # Hz
    'VOCAB_SIZE': 128,
    'DEFAULT_SEQ_LENGTH': 25,
    'DEFAULT_BATCH_SIZE': 64,
    'DEFAULT_EPOCHS': 50,
    'DATA_URL': 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
    'DATA_DIR': pathlib.Path('data/maestro-v2.0.0'),
    'CHECKPOINT_DIR': pathlib.Path('./training_checkpoints'),
    'MODEL_PATH': pathlib.Path('./saved_model'),
    'CACHE_DIR': './cache_dir',
    'NUM_TRACKS': 3,
    'NUM_PREDICTIONS': 120,
    'TEMPERATURE': 1.0,
    'LEARNING_RATE': 0.005,
    'BUFFER_MULTIPLIER': 1.0, # Factor to adjust buffer size
    'ADVANCED_MODEL_SCALE': 10, # Scale factor for advanced model
}

# Create necessary directories
CONFIG['DATA_DIR'].mkdir(parents=True, exist_ok=True)
CONFIG['CHECKPOINT_DIR'].mkdir(parents=True, exist_ok=True)
CONFIG['MODEL_PATH'].mkdir(parents=True, exist_ok=True)
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# DATA HANDLING FUNCTIONS
# =============================================================================

def download_dataset():
    if CONFIG['DATA_DIR'].exists() and any(CONFIG['DATA_DIR'].iterdir()):
        logging.info("MAESTRO dataset already exists. Skipping download.")
        return
    logging.info("Downloading and extracting MAESTRO dataset...")
    try:
        tf.keras.utils.get_file(
            fname='maestro-v2.0.0-midi.zip',
            origin=CONFIG['DATA_URL'],
            extract=True,
            cache_dir='.',
            cache_subdir='data'
        )
        if not CONFIG['DATA_DIR'].exists() or not any(CONFIG['DATA_DIR'].iterdir()):
            raise FileNotFoundError("Dataset extraction failed or directory is empty.")
        logging.info("Download and extraction completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while downloading or extracting the dataset: {e}")
        sys.exit(1)

def list_midi_files():
    search_dir = CONFIG['DATA_DIR'].parent / 'maestro-v2.0.0'
    filenames = glob.glob(str(search_dir / '**/*.mid*'), recursive=True)
    if not filenames:
        logging.error("No MIDI files found. Please check if the dataset was downloaded correctly.")
        logging.error(f"Searched in directory: {search_dir}")
        sys.exit(1)
    logging.info(f'Number of MIDI files found: {len(filenames)}')
    return filenames

def parse_midi_file(midi_file):
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        if not pm.instruments:
            logging.warning(f"No instruments found in {midi_file}. Skipping.")
            return None
        return pm
    except Exception as e:
        logging.error(f"Error parsing {midi_file}: {e}")
        return None

def extract_notes(midi_file):
    pm = parse_midi_file(midi_file)
    if pm is None or not pm.instruments:
        logging.warning(f"Skipping file {midi_file} due to missing instruments.")
        return pd.DataFrame()
    instrument = pm.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    if not sorted_notes:
        logging.warning(f"No notes found in {midi_file}. Skipping.")
        return pd.DataFrame()
    notes = collections.defaultdict(list)
    prev_start = sorted_notes[0].start
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)
        prev_start = start
    return pd.DataFrame({key: np.array(val) for key, val in notes.items()})

def notes_to_midi(notes_df, out_file, instrument_name='Acoustic Grand Piano', velocity=100):
    pm = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    prev_start = 0
    for _, note in notes_df.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']), start=start, end=end)
        instrument.notes.append(midi_note)
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    logging.info(f"MIDI file saved to {out_file}")
    return pm

# =============================================================================
# DATA VISUALIZATION FUNCTIONS
# =============================================================================

def plot_piano_roll(notes, count=None, title_suffix=''):
    count = count or len(notes['pitch'])
    title = f'First {count} notes {title_suffix}' if count else f'Whole track {title_suffix}'
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    plt.title(title)
    plt.show()

def plot_distributions(notes, drop_percentile=2.5, title_suffix=''):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20, kde=True, color='skyblue')
    plt.title('Pitch Distribution')
    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21), kde=True, color='salmon')
    plt.title('Step Distribution')
    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21), kde=True, color='lightgreen')
    plt.title('Duration Distribution')
    plt.suptitle(title_suffix)
    plt.tight_layout()
    plt.show()

# =============================================================================
# SEQUENCE CREATION & MODEL COMPONENTS
# =============================================================================

def create_sequences(dataset, seq_length, vocab_size):
    seq_length += 1 # Extra element for label
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        return x / [vocab_size, 1.0, 1.0]

    def split_labels(seq_batch):
        inputs = seq_batch[:-1]
        labels_dense = seq_batch[-1]
        labels = {
            'pitch': tf.cast(labels_dense[0], tf.int32),
            'step': labels_dense[1],
            'duration': labels_dense[2],
        }
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

@tf.keras.utils.register_keras_serializable(package="Custom")
def mse_with_positive_pressure(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    positive_pressure = 10 * tf.reduce_mean(tf.maximum(-y_pred, 0.0))
    return mse + positive_pressure

def advanced_build_model(seq_length, learning_rate, scale_factor):
    # Increase complexity by adding multiple LSTM layers and Dense layers
    inputs = tf.keras.Input(shape=(seq_length, 3), name='input_notes')
    x = inputs
    for i in range(3): # Add 3 stacked LSTM layers
        units = 128 * scale_factor # Scale units by factor
        return_sequences = (i < 2) # Only last layer does not return sequences
        x = tf.keras.layers.LSTM(units, return_sequences=return_sequences,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 name=f'lstm_layer_{i+1}')(x)
    # Add additional Dense layers for more nuance
    dense_out = tf.keras.layers.Dense(256 * scale_factor, activation='relu', name='dense_1')(x)
    dense_out = tf.keras.layers.Dense(128 * scale_factor, activation='relu', name='dense_2')(dense_out)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(dense_out),
        'step': tf.keras.layers.Dense(1, name='step')(dense_out),
        'duration': tf.keras.layers.Dense(1, name='duration')(dense_out),
    }
    model = tf.keras.Model(inputs, outputs, name='advanced_music_rnn_model')

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         clipnorm=1.0) # Use gradient clipping
    model.compile(loss=loss,
                  loss_weights={'pitch': 0.05, 'step': 1.0, 'duration': 1.0},
                  optimizer=optimizer)
    logging.info("Advanced Model Summary:")
    model.summary(print_fn=logging.info)
    return model

# =============================================================================
# TRAINING AND GENERATION FUNCTIONS
# =============================================================================

def train_model(model, train_ds, epochs, checkpoint_dir, steps_per_epoch):
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        logging.info(f"Resuming training from checkpoint: {latest_ckpt}")
        model.load_weights(latest_ckpt)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'ckpt_{epoch}.keras'),
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    logging.info("Starting model training...")
    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    logging.info("Model training completed.")
    return history

def generate_next_note(notes, model, temperature):
    assert temperature > 0, "Temperature must be greater than 0."
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs, verbose=0)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1).numpy()[0]
    duration = tf.squeeze(duration, axis=-1).numpy()[0]
    step = tf.squeeze(step, axis=-1).numpy()[0]
    step = max(0, step)
    duration = max(0, duration)
    return int(pitch), float(step), float(duration)

def generate_music(model, initial_notes, num_predictions, temperature, out_file, instrument_name):
    generated_notes = []
    input_notes = initial_notes.copy()
    prev_start = input_notes[-1][2]
    for _ in range(num_predictions):
        pitch, step, dur = generate_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + dur
        generated_notes.append({'pitch': pitch, 'step': step, 'duration': dur, 'start': start, 'end': end})
        next_input = (pitch, step, dur)
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(next_input, 0), axis=0)
        prev_start = start
    generated_df = pd.DataFrame(generated_notes)
    midi_pm = notes_to_midi(generated_df, out_file, instrument_name)
    logging.info(f"Generated {num_predictions} notes and saved to {out_file}")
    return midi_pm, generated_df

def visualize_generated_notes(generated_df):
    plot_piano_roll(generated_df, title_suffix='(Generated)')
    plot_distributions(generated_df, title_suffix='(Generated)')

# =============================================================================
# AUTOMATIC SETUP AND MAIN PIPELINE
# =============================================================================

def automatic_setup():
    download_dataset()
    if not os.path.exists('all_notes.csv'):
        filenames = list_midi_files()
        all_notes = []
        for idx, f in enumerate(filenames, 1):
            logging.info(f"Processing file {idx}/{len(filenames)}: {f}")
            notes_df = extract_notes(f)
            if not notes_df.empty:
                all_notes.append(notes_df)
        if all_notes:
            all_notes_df = pd.concat(all_notes, ignore_index=True)
            all_notes_df.to_csv('all_notes.csv', index=False)
            logging.info(f"Extracted and saved notes from {len(all_notes)} files.")
        else:
            logging.error("No notes were extracted from the MIDI files.")
            sys.exit(1)
    else:
        logging.info("all_notes.csv found, skipping MIDI extraction.")

    all_df = pd.read_csv('all_notes.csv')
    n_notes = len(all_df)
    logging.info(f'Number of notes parsed: {n_notes}')

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_df[key] for key in key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    seq_ds = create_sequences(notes_ds, CONFIG['DEFAULT_SEQ_LENGTH'], CONFIG['VOCAB_SIZE'])
    buffer_size = int(min(n_notes - CONFIG['DEFAULT_SEQ_LENGTH'], 200000) * CONFIG['BUFFER_MULTIPLIER'])
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .take(buffer_size) # Take a subset before caching to fix warnings
                .cache(CONFIG['CACHE_DIR'])
                .repeat()
                .batch(CONFIG['DEFAULT_BATCH_SIZE'], drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    # Calculate steps_per_epoch based on buffer_size and batch_size
    steps_per_epoch = buffer_size // CONFIG['DEFAULT_BATCH_SIZE']

    model = advanced_build_model(CONFIG['DEFAULT_SEQ_LENGTH'], CONFIG['LEARNING_RATE'], CONFIG['ADVANCED_MODEL_SCALE'])
    train_model(model, train_ds, CONFIG['DEFAULT_EPOCHS'], CONFIG['CHECKPOINT_DIR'], steps_per_epoch)
    model_save_path = CONFIG['MODEL_PATH'] / 'final_model.keras'
    model.save(str(model_save_path))
    logging.info(f"Model saved successfully at {model_save_path}")

def main():
    automatic_setup()

    model_files = list(CONFIG['MODEL_PATH'].glob('*.keras'))
    if not model_files:
        logging.error("No model found after training.")
        sys.exit(1)
    model_file = model_files[0]
    model = tf.keras.models.load_model(
        str(model_file),
        custom_objects={'mse_with_positive_pressure': mse_with_positive_pressure}
    )
    logging.info("Model loaded for generation.")

    if not os.path.exists('all_notes.csv'):
        logging.info("all_notes.csv not found. Running automatic setup...")
        automatic_setup()

    all_notes_df = pd.read_csv('all_notes.csv')
    sample_notes = all_notes_df[['pitch', 'step', 'duration']].values
    if len(sample_notes) < CONFIG['DEFAULT_SEQ_LENGTH']:
        logging.error(f"Not enough initial notes. Found {len(sample_notes)} total notes.")
        sys.exit(1)
    initial_notes = sample_notes[:CONFIG['DEFAULT_SEQ_LENGTH']] / np.array([CONFIG['VOCAB_SIZE'], 1, 1])

    for track in range(1, CONFIG['NUM_TRACKS'] + 1):
        out_file = f'output_track_{track}.mid'
        instrument = 'Acoustic Grand Piano'
        logging.info(f"Generating music for track {track}...")
        generate_music(model, initial_notes, CONFIG['NUM_PREDICTIONS'], CONFIG['TEMPERATURE'], out_file, instrument)
        logging.info(f"Track {track} generated and saved to {out_file}.")

if __name__ == '__main__':
    main()

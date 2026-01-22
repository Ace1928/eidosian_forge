import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import wave
import contextlib
import srt
import datetime


def convert_mp3_to_wav(mp3_file_path: str, wav_file_path: str):
    sound = AudioSegment.from_mp3(mp3_file_path)
    sound.export(wav_file_path, format="wav")


def remove_vocals(input_file_path: str, output_file_path: str):
    y, sr = librosa.load(input_file_path, sr=None, mono=False)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(
        S_full,
        aggregate=np.median,
        metric="cosine",
        width=int(librosa.time_to_frames(2, sr=sr)),
    )
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(
        S_filter, margin_i * (S_full - S_filter), power=power
    )
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    librosa.output.write_wav(output_file_path, librosa.istft(S_background * phase), sr)


def extract_lyrics(input_file_path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(input_file_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "No lyrics could be recognized."
    except sr.RequestError as e:
        return f"Could not request results; {e}"


def get_audio_duration(wav_file_path: str) -> float:
    with contextlib.closing(wave.open(wav_file_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def create_subtitles(lyrics: str, duration: float) -> str:
    segments = lyrics.split()
    num_segments = len(segments)
    segment_duration = duration / num_segments
    subs = []
    for i, segment in enumerate(segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        subs.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=start_time),
                end=datetime.timedelta(seconds=end_time),
                content=segment,
            )
        )
    return srt.compose(subs)


# Example usage:
mp3_file_path = "path/to/input_song.mp3"
wav_file_path = "path/to/output_song.wav"
vocal_removed_file_path = "path/to/vocal_removed.wav"
subtitle_file_path = "path/to/subtitles.srt"

convert_mp3_to_wav(mp3_file_path, wav_file_path)
remove_vocals(wav_file_path, vocal_removed_file_path)
lyrics = extract_lyrics(wav_file_path)
duration = get_audio_duration(vocal_removed_file_path)
subtitles = create_subtitles(lyrics, duration)

with open(subtitle_file_path, "w") as f:
    f.write(subtitles)

print("Karaoke track and subtitles created successfully!")

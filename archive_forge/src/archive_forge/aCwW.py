import speech_recognition as sr
import subprocess
import os

# Create a recognizer object
recognizer = sr.Recognizer()

# Define the path for the audio file
audio_path = "/home/lloyd/Downloads/audio/output.mp3"
wav_audio_path = "/home/lloyd/Downloads/audio/output.wav"

# Convert mp3 to wav if necessary
if audio_path.endswith(".mp3"):
    subprocess.run(["ffmpeg", "-i", audio_path, wav_audio_path])
    audio_path = wav_audio_path

# Load the audio file
with sr.AudioFile(audio_path) as source:
    # Read the audio data
    audio = recognizer.record(source)

# Perform speech recognition
try:
    text = recognizer.recognize_google(audio)
    print("Transcription:", text)
except sr.UnknownValueError:
    print("Speech recognition could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results from the speech recognition service; {e}")

# Clean up the converted file
if audio_path == wav_audio_path:
    os.remove(wav_audio_path)

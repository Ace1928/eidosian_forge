from gtts import gTTS
import os

# Text file to convert
text_file = "example.txt"

# Output audio file
audio_file = "output.mp3"

# Read the text from the file
with open(text_file, "r") as file:
    text = file.read()

# Create a gTTS object and generate the audio
tts = gTTS(text=text, lang="en")
tts.save(audio_file)

print(f"Text converted to speech. Audio saved as {audio_file}")

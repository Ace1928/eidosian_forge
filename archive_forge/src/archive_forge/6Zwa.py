from gtts import gTTS
import os

# Text to convert to speech
text = "Hello, this is an example of text-to-speech conversion using Python."

# Create a gTTS object
tts = gTTS(text=text, lang="en")

# Save the speech as an audio file
tts.save("output.mp3")

# Play the audio file
os.system("start output.mp3")

from langdetect import detect

# Text to detect the language
text = "Bonjour, comment allez-vous?"

# Detect the language
language = detect(text)

# Print the detected language
print("Detected Language:", language)

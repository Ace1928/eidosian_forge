from googletrans import Translator

# Text to translate
text = "Bonjour, comment allez-vous?"

# Create a translator object
translator = Translator()

# Translate the text
translation = translator.translate(text, dest="en")

# Print the translated text
print("Original Text:", text)
print("Translated Text:", translation.text)

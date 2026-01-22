from googletrans import Translator, LANGUAGES

# Text to translate
text = "Bonjour, comment allez-vous?"

# Check if the desired language is available
if "en" in LANGUAGES.values():
    # Create a translator object
    translator = Translator()

    # Translate the text
    translation = translator.translate(text, dest="en")

    # Print the translated text
    print("Original Text:", text)
    print("Translated Text:", translation.text)
else:
    print("English translation is not supported.")

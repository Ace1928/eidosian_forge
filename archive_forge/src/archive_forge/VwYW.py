import argostranslate.package
import argostranslate.translate
import os

# Text to translate
text = "Bonjour, comment allez-vous?"

# Define the local directory to store packages
local_package_dir = "/home/lloyd/Downloads/argos/packages"

# Ensure the latest packages are available and install necessary language packages
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()


# Download all available translation packages to a local directory if not already present
def download_all_packages():
    for package in available_packages:
        local_package_path = os.path.join(
            local_package_dir, f"{package.from_code}_to_{package.to_code}.argosmodel"
        )
        package.download()


download_all_packages()

# Load packages from the local directory
argostranslate.package.load_packages(local_package_dir)

# Load installed languages
installed_languages = argostranslate.translate.get_installed_languages()
language_codes = {lang.code: lang for lang in installed_languages}


# Detect the language of the input text
def detect_language(text):
    for lang in installed_languages:
        if lang.can_translate_from(text):
            return lang
    return None


source_lang = detect_language(text)

# Assuming English is installed
to_lang = next((lang for lang in installed_languages if lang.code == "en"), None)

try:
    # Attempt to translate directly to English and infer the source language
    if source_lang and to_lang:
        translation = source_lang.translate_to(text, to_lang)
        print("Original Text:", text)
        print("Translated Text:", translation)
    else:
        print("Translation to English not supported or source language not detected.")
except Exception as e:
    print(f"An error occurred during translation: {e}")

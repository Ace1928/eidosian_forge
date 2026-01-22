import argostranslate.package, argostranslate.translate
from langdetect import detect
import logging
import pathlib

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Text to translate
text = "Bonjour, comment allez-vous?"


# Function to robustly detect language
def robust_detect_language(text: str) -> str:
    try:
        detected_language = detect(text)
        logging.info(f"Detected Language: {detected_language}")
        return detected_language
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return None


# Ensure the package index is up-to-date and get available packages
def update_and_load_packages():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    installed_languages = argostranslate.translate.get_installed_languages()
    language_codes = {lang.code: lang for lang in installed_languages}
    return available_packages, installed_languages, language_codes


# Download and verify translation packages
def download_and_verify_package(source_lang_code: str, target_lang_code: str) -> bool:
    available_packages, _, _ = update_and_load_packages()
    desired_package = next(
        (
            pkg
            for pkg in available_packages
            if pkg.from_code == source_lang_code and pkg.to_code == target_lang_code
        ),
        None,
    )
    if desired_package:
        download_path = desired_package.download()
        argostranslate.package.install_from_path(pathlib.Path(download_path))
        logging.info(f"Package downloaded and installed from {download_path}")
        return True
    else:
        logging.error(
            f"No available package from {source_lang_code} to {target_lang_code}"
        )
        return False


# Enhanced language detection and translation
def translate_text(text: str, target_lang_code="en"):
    detected_language = robust_detect_language(text)
    if detected_language:
        _, installed_languages, language_codes = update_and_load_packages()
        if detected_language not in language_codes:
            if not download_and_verify_package(detected_language, target_lang_code):
                logging.error(
                    f"No available translation package from {detected_language} to {target_lang_code}."
                )
                return
        translation = language_codes[detected_language].get_translation(
            language_codes[target_lang_code]
        )
        translated_text = translation.translate(text)
        logging.info(f"Original Text: {text}")
        logging.info(f"Translated Text: {translated_text}")


# Translate text
translate_text(text)

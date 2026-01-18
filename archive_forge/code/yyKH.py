import argostranslate.package
import argostranslate.translate

# Text to translate
text = "Bonjour, comment allez-vous?"

# Ensure the latest packages are available and install necessary language packages
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()


# Install packages for detected language to English if not already installed
def install_language_package(from_code, to_code="en"):
    package_to_install = next(
        (
            pkg
            for pkg in available_packages
            if pkg.from_code == from_code and pkg.to_code == to_code
        ),
        None,
    )
    if package_to_install:
        argostranslate.package.install_from_path(package_to_install.download())
    else:
        print(f"No available package to translate from {from_code} to {to_code}")


# Detect language and translate
installed_languages = argostranslate.translate.get_installed_languages()
language_codes = {lang.code: lang for lang in installed_languages}

try:
    # Detect the language of the input text
    detected_language = argostranslate.translate.detect(text)
    detected_language_code = detected_language.language.code

    if detected_language_code in language_codes:
        from_lang = language_codes[detected_language_code]
        to_lang = language_codes.get("en", None)

        if not to_lang:
            install_language_package(detected_language_code, "en")
            # Reload installed languages after installation
            installed_languages = argostranslate.translate.get_installed_languages()
            to_lang = next(lang for lang in installed_languages if lang.code == "en")

        translation = from_lang.get_translation(to_lang)
        translated_text = translation.translate(text)
        print("Original Text:", text)
        print("Translated Text:", translated_text)
    else:
        print(f"Translation from {detected_language_code} to English is not supported.")
except Exception as e:
    print(f"An error occurred during translation: {e}")

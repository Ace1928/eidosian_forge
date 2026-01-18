import pytest
from charset_normalizer.cd import encoding_languages, mb_encoding_languages, is_multi_byte_encoding, get_target_features
@pytest.mark.parametrize('language, expected_have_accents, expected_pure_latin', [('English', False, True), ('French', True, True), ('Hebrew', False, False), ('Arabic', False, False), ('Vietnamese', True, True), ('Turkish', True, True)])
def test_target_features(language, expected_have_accents, expected_pure_latin):
    target_have_accents, target_pure_latin = get_target_features(language)
    assert target_have_accents is expected_have_accents
    assert target_pure_latin is expected_pure_latin
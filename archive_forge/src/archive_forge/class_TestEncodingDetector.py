import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
class TestEncodingDetector(object):

    def test_encoding_detector_replaces_junk_in_encoding_name_with_replacement_character(self):
        detected = EncodingDetector(b'<?xml version="1.0" encoding="UTF-\xdb" ?>')
        encodings = list(detected.encodings)
        assert 'utf-�' in encodings

    def test_detect_html5_style_meta_tag(self):
        for data in (b'<html><meta charset="euc-jp" /></html>', b"<html><meta charset='euc-jp' /></html>", b'<html><meta charset=euc-jp /></html>', b'<html><meta charset=euc-jp/></html>'):
            dammit = UnicodeDammit(data, is_html=True)
            assert 'euc-jp' == dammit.original_encoding

    def test_last_ditch_entity_replacement(self):
        doc = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\n<html><b>\xd8\xa8\xd8\xaa\xd8\xb1</b>\n<i>\xc8\xd2\xd1\x90\xca\xd1\xed\xe4</i></html>'
        chardet = bs4.dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:

            def noop(str):
                return None
            bs4.dammit.chardet_dammit = noop
            dammit = UnicodeDammit(doc)
            assert True == dammit.contains_replacement_characters
            assert '�' in dammit.unicode_markup
            soup = BeautifulSoup(doc, 'html.parser')
            assert soup.contains_replacement_characters
        finally:
            logging.disable(logging.NOTSET)
            bs4.dammit.chardet_dammit = chardet

    def test_byte_order_mark_removed(self):
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)
        assert '<a>áé</a>' == dammit.unicode_markup
        assert 'utf-16le' == dammit.original_encoding

    def test_known_definite_versus_user_encodings(self):
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)
        before = UnicodeDammit(data, known_definite_encodings=['utf-16'])
        assert 'utf-16' == before.original_encoding
        after = UnicodeDammit(data, user_encodings=['utf-8'])
        assert 'utf-16le' == after.original_encoding
        assert ['utf-16le'] == [x[0] for x in dammit.tried_encodings]
        hebrew = b'\xed\xe5\xec\xf9'
        dammit = UnicodeDammit(hebrew, known_definite_encodings=['utf-8'], user_encodings=['iso-8859-8'])
        assert 'iso-8859-8' == dammit.original_encoding
        assert ['utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]

    def test_deprecated_override_encodings(self):
        hebrew = b'\xed\xe5\xec\xf9'
        dammit = UnicodeDammit(hebrew, known_definite_encodings=['shift-jis'], override_encodings=['utf-8'], user_encodings=['iso-8859-8'])
        assert 'iso-8859-8' == dammit.original_encoding
        assert ['shift-jis', 'utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]

    def test_detwingle(self):
        utf8 = ('☃' * 3).encode('utf8')
        windows_1252 = '“Hi, I like Windows!”'.encode('windows_1252')
        doc = utf8 + windows_1252 + utf8
        with pytest.raises(UnicodeDecodeError):
            doc.decode('utf8')
        fixed = UnicodeDammit.detwingle(doc)
        assert '☃☃☃“Hi, I like Windows!”☃☃☃' == fixed.decode('utf8')

    def test_detwingle_ignores_multibyte_characters(self):
        for tricky_unicode_char in ('œ', 'ₓ', 'ð\x90\x90\x93'):
            input = tricky_unicode_char.encode('utf8')
            assert input.endswith(b'\x93')
            output = UnicodeDammit.detwingle(input)
            assert output == input

    def test_find_declared_encoding(self):
        html_unicode = '<html><head><meta charset="utf-8"></head></html>'
        html_bytes = html_unicode.encode('ascii')
        xml_unicode = '<?xml version="1.0" encoding="ISO-8859-1" ?>'
        xml_bytes = xml_unicode.encode('ascii')
        m = EncodingDetector.find_declared_encoding
        assert m(html_unicode, is_html=False) is None
        assert 'utf-8' == m(html_unicode, is_html=True)
        assert 'utf-8' == m(html_bytes, is_html=True)
        assert 'iso-8859-1' == m(xml_unicode)
        assert 'iso-8859-1' == m(xml_bytes)
        spacer = b' ' * 5000
        assert m(spacer + html_bytes) is None
        assert m(spacer + xml_bytes) is None
        assert m(spacer + html_bytes, is_html=True, search_entire_document=True) == 'utf-8'
        assert m(xml_bytes, search_entire_document=True) == 'iso-8859-1'
        assert m(b' ' + xml_bytes, search_entire_document=True) == 'iso-8859-1'
        assert m(b'a' + xml_bytes, search_entire_document=True) is None
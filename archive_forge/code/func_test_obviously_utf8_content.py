from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
@pytest.mark.parametrize('payload', ['ȍ\x1b'.encode('utf-8'), 'héllo world!\n'.encode('utf_8'), '我没有埋怨，磋砣的只是一些时间。'.encode('utf_8'), 'Bсеки човек има право на образование. Oбразованието трябва да бъде безплатно, поне що се отнася до началното и основното образование.'.encode('utf_8'), 'Bсеки човек има право на образование.'.encode('utf_8'), '(° ͜ʖ °), creepy face, smiley 😀'.encode('utf_8'), '["Financiën", "La France"]'.encode('utf_8'), "Qu'est ce que une étoile?".encode('utf_8'), '<?xml ?><c>Financiën</c>'.encode('utf_8'), '😀'.encode('utf_8')])
def test_obviously_utf8_content(payload):
    best_guess = from_bytes(payload).best()
    assert best_guess is not None, 'Dead-simple UTF-8 detection has failed!'
    assert best_guess.encoding == 'utf_8', 'Dead-simple UTF-8 detection is wrongly detected!'
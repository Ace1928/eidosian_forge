import pytest
from spacy.lang.tl.lex_attrs import like_num
def test_tl_tokenizer_handles_long_text(tl_tokenizer):
    text = '\n    Tingin tayo nang tingin. Kailangan lamang nating dumilat at\n    marami tayong makikita. At ang pagtingin ay isang gawain na ako lamang ang\n    makagagawa, kung ako nga ang makakita. Kahit na napanood na ng aking\n    matalik na kaibigan ang isang sine, kailangan ko pa ring panoorin, kung\n    ako nga ang may gustong makakita. Kahit na gaano kadikit ang aming\n    pagkabuklod, hindi siya maaaring tumingin sa isang paraan na ako ang\n    nakakakita. Kung ako ang makakita, ako lamang ang makatitingin.\n    '
    tokens = tl_tokenizer(text)
    assert len(tokens) == 97
import sys
import pytest
def test_tokenizer_handles_emoticons(tokenizer):
    text = ":o :/ :'( >:o (: :) >.< XD -__- o.O ;D :-) @_@ :P 8D :1 >:( :D =| :> ...."
    tokens = tokenizer(text)
    assert tokens[0].text == ':o'
    assert tokens[1].text == ':/'
    assert tokens[2].text == ":'("
    assert tokens[3].text == '>:o'
    assert tokens[4].text == '(:'
    assert tokens[5].text == ':)'
    assert tokens[6].text == '>.<'
    assert tokens[7].text == 'XD'
    assert tokens[8].text == '-__-'
    assert tokens[9].text == 'o.O'
    assert tokens[10].text == ';D'
    assert tokens[11].text == ':-)'
    assert tokens[12].text == '@_@'
    assert tokens[13].text == ':P'
    assert tokens[14].text == '8D'
    assert tokens[15].text == ':1'
    assert tokens[16].text == '>:('
    assert tokens[17].text == ':D'
    assert tokens[18].text == '=|'
    assert tokens[19].text == ':>'
    assert tokens[20].text == '....'
import pytest
import math
import rpy2.rinterface as ri
def test_r_to_NACharacter():
    na_character = ri.NA_Character
    r_na_character = ri.evalr('NA_character_')
    assert r_na_character.typeof == ri.RTYPES.STRSXP
    assert len(r_na_character) == 1
    assert r_na_character.get_charsxp(0).rid == na_character.rid
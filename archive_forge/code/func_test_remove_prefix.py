import sys
import pytest
@pytest.mark.parametrize('string, prefix, expected', (('wildcat', 'wild', 'cat'), ('blackbird', 'black', 'bird'), ('housefly', 'house', 'fly'), ('ladybug', 'lady', 'bug'), ('rattlesnake', 'rattle', 'snake'), ('baboon', 'badger', 'baboon'), ('quetzal', 'elk', 'quetzal')))
def test_remove_prefix(string, prefix, expected):
    result = removeprefix(string, prefix)
    assert result == expected
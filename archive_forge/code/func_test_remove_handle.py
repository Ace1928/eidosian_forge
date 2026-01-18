from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_remove_handle(self):
    """
        Test remove_handle() from casual.py with specially crafted edge cases
        """
    tokenizer = TweetTokenizer(strip_handles=True)
    test1 = '@twitter hello @twi_tter_. hi @12345 @123news'
    expected = ['hello', '.', 'hi']
    result = tokenizer.tokenize(test1)
    assert result == expected
    test2 = '@n`@n~@n(@n)@n-@n=@n+@n\\@n|@n[@n]@n{@n}@n;@n:@n\'@n"@n/@n?@n.@n,@n<@n>@n @n\n@n ñ@n.ü@n.ç@n.'
    expected = ['`', '~', '(', ')', '-', '=', '+', '\\', '|', '[', ']', '{', '}', ';', ':', "'", '"', '/', '?', '.', ',', '<', '>', 'ñ', '.', 'ü', '.', 'ç', '.']
    result = tokenizer.tokenize(test2)
    assert result == expected
    test3 = 'a@n j@n z@n A@n L@n Z@n 1@n 4@n 7@n 9@n 0@n _@n !@n @@n #@n $@n %@n &@n *@n'
    expected = ['a', '@n', 'j', '@n', 'z', '@n', 'A', '@n', 'L', '@n', 'Z', '@n', '1', '@n', '4', '@n', '7', '@n', '9', '@n', '0', '@n', '_', '@n', '!', '@n', '@', '@n', '#', '@n', '$', '@n', '%', '@n', '&', '@n', '*', '@n']
    result = tokenizer.tokenize(test3)
    assert result == expected
    test4 = '@n!a @n#a @n$a @n%a @n&a @n*a'
    expected = ['!', 'a', '#', 'a', '$', 'a', '%', 'a', '&', 'a', '*', 'a']
    result = tokenizer.tokenize(test4)
    assert result == expected
    test5 = '@n!@n @n#@n @n$@n @n%@n @n&@n @n*@n @n@n @@n @n@@n @n_@n @n7@n @nj@n'
    expected = ['!', '@n', '#', '@n', '$', '@n', '%', '@n', '&', '@n', '*', '@n', '@n', '@n', '@', '@n', '@n', '@', '@n', '@n_', '@n', '@n7', '@n', '@nj', '@n']
    result = tokenizer.tokenize(test5)
    assert result == expected
    test6 = '@abcdefghijklmnopqrstuvwxyz @abcdefghijklmno1234 @abcdefghijklmno_ @abcdefghijklmnoendofhandle'
    expected = ['pqrstuvwxyz', '1234', '_', 'endofhandle']
    result = tokenizer.tokenize(test6)
    assert result == expected
    test7 = '@abcdefghijklmnop@abcde @abcdefghijklmno@abcde @abcdefghijklmno_@abcde @abcdefghijklmno5@abcde'
    expected = ['p', '@abcde', '@abcdefghijklmno', '@abcde', '_', '@abcde', '5', '@abcde']
    result = tokenizer.tokenize(test7)
    assert result == expected
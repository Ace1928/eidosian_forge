from io import BytesIO
from unittest import TestCase
from ..mailmap import Mailmap, read_mailmap
class ReadMailmapTests(TestCase):

    def test_read(self):
        b = BytesIO(b'Jane Doe         <jane@desktop.(none)>\nJoe R. Developer <joe@example.com>\n# A comment\n<cto@company.xx>                       <cto@coompany.xx> # Comment\nSome Dude <some@dude.xx>         nick1 <bugs@company.xx>\nOther Author <other@author.xx>   nick2 <bugs@company.xx>\nOther Author <other@author.xx>         <nick2@company.xx>\nSanta Claus <santa.claus@northpole.xx> <me@company.xx>\n')
        self.assertEqual([((b'Jane Doe', b'jane@desktop.(none)'), None), ((b'Joe R. Developer', b'joe@example.com'), None), ((None, b'cto@company.xx'), (None, b'cto@coompany.xx')), ((b'Some Dude', b'some@dude.xx'), (b'nick1', b'bugs@company.xx')), ((b'Other Author', b'other@author.xx'), (b'nick2', b'bugs@company.xx')), ((b'Other Author', b'other@author.xx'), (None, b'nick2@company.xx')), ((b'Santa Claus', b'santa.claus@northpole.xx'), (None, b'me@company.xx'))], list(read_mailmap(b)))
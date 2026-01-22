from lxml import etree
import sys
import re
import doctest
class LHTMLOutputChecker(LXMLOutputChecker):

    def get_default_parser(self):
        return html_fromstring
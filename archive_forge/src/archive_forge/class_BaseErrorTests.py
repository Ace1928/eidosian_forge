from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
class BaseErrorTests(unittest.TestCase):

    def test_getElementPlain(self) -> None:
        """
        Test getting an element for a plain error.
        """
        e = error.BaseError('feature-not-implemented')
        element = e.getElement()
        self.assertIdentical(element.uri, None)
        self.assertEqual(len(element.children), 1)

    def test_getElementText(self) -> None:
        """
        Test getting an element for an error with a text.
        """
        e = error.BaseError('feature-not-implemented', 'text')
        element = e.getElement()
        self.assertEqual(len(element.children), 2)
        self.assertEqual(str(element.text), 'text')
        self.assertEqual(element.text.getAttribute((NS_XML, 'lang')), None)

    def test_getElementTextLang(self) -> None:
        """
        Test getting an element for an error with a text and language.
        """
        e = error.BaseError('feature-not-implemented', 'text', 'en_US')
        element = e.getElement()
        self.assertEqual(len(element.children), 2)
        self.assertEqual(str(element.text), 'text')
        self.assertEqual(element.text[NS_XML, 'lang'], 'en_US')

    def test_getElementAppCondition(self) -> None:
        """
        Test getting an element for an error with an app specific condition.
        """
        ac = domish.Element(('testns', 'myerror'))
        e = error.BaseError('feature-not-implemented', appCondition=ac)
        element = e.getElement()
        self.assertEqual(len(element.children), 2)
        self.assertEqual(element.myerror, ac)
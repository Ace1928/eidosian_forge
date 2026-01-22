from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
class ExceptionFromStanzaTests(unittest.TestCase):

    def test_basic(self) -> None:
        """
        Test basic operations of exceptionFromStanza.

        Given a realistic stanza, check if a sane exception is returned.

        Using this stanza::

          <iq type='error'
              from='pubsub.shakespeare.lit'
              to='francisco@denmark.lit/barracks'
              id='subscriptions1'>
            <pubsub xmlns='http://jabber.org/protocol/pubsub'>
              <subscriptions/>
            </pubsub>
            <error type='cancel'>
              <feature-not-implemented
                xmlns='urn:ietf:params:xml:ns:xmpp-stanzas'/>
              <unsupported xmlns='http://jabber.org/protocol/pubsub#errors'
                           feature='retrieve-subscriptions'/>
            </error>
          </iq>
        """
        stanza = domish.Element((None, 'stanza'))
        p = stanza.addElement(('http://jabber.org/protocol/pubsub', 'pubsub'))
        p.addElement('subscriptions')
        e = stanza.addElement('error')
        e['type'] = 'cancel'
        e.addElement((NS_XMPP_STANZAS, 'feature-not-implemented'))
        uc = e.addElement(('http://jabber.org/protocol/pubsub#errors', 'unsupported'))
        uc['feature'] = 'retrieve-subscriptions'
        result = error.exceptionFromStanza(stanza)
        self.assertIsInstance(result, error.StanzaError)
        self.assertEqual('feature-not-implemented', result.condition)
        self.assertEqual('cancel', result.type)
        self.assertEqual(uc, result.appCondition)
        self.assertEqual([p], result.children)

    def test_legacy(self) -> None:
        """
        Test legacy operations of exceptionFromStanza.

        Given a realistic stanza with only legacy (pre-XMPP) error information,
        check if a sane exception is returned.

        Using this stanza::

          <message type='error'
                   to='piers@pipetree.com/Home'
                   from='qmacro@jaber.org'>
            <body>Are you there?</body>
            <error code='502'>Unable to resolve hostname.</error>
          </message>
        """
        stanza = domish.Element((None, 'stanza'))
        p = stanza.addElement('body', content='Are you there?')
        e = stanza.addElement('error', content='Unable to resolve hostname.')
        e['code'] = '502'
        result = error.exceptionFromStanza(stanza)
        self.assertIsInstance(result, error.StanzaError)
        self.assertEqual('service-unavailable', result.condition)
        self.assertEqual('wait', result.type)
        self.assertEqual('Unable to resolve hostname.', result.text)
        self.assertEqual([p], result.children)
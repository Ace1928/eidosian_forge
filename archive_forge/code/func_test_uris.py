import urlparse
def test_uris(self):
    """Test that URIs are invariant under the transformation."""
    invariant = [u'ftp://ftp.is.co.za/rfc/rfc1808.txt', u'http://www.ietf.org/rfc/rfc2396.txt', u'ldap://[2001:db8::7]/c=GB?objectClass?one', u'mailto:John.Doe@example.com', u'news:comp.infosystems.www.servers.unix', u'tel:+1-816-555-1212', u'telnet://192.0.2.16:80/', u'urn:oasis:names:specification:docbook:dtd:xml:4.1.2']
    for uri in invariant:
        self.assertEqual(uri, iri2uri(uri))
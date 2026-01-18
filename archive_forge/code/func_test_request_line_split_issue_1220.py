import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_request_line_split_issue_1220(test_client):
    """Check that HTTP request line of exactly 256 chars length is OK."""
    Request_URI = '/hello?intervenant-entreprise-evenement_classaction=evenement-mailremerciements&_path=intervenant-entreprise-evenement&intervenant-entreprise-evenement_action-id=19404&intervenant-entreprise-evenement_id=19404&intervenant-entreprise_id=28092'
    assert len('GET %s HTTP/1.1\r\n' % Request_URI) == 256
    actual_resp_body = test_client.get(Request_URI)[2]
    assert actual_resp_body == b'Hello world!'
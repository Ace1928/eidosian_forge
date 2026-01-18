import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def verify_domain_dkim(self, domain):
    """
        Returns a set of DNS records, or tokens, that must be published in the
        domain name's DNS to complete the DKIM verification process. These
        tokens are DNS ``CNAME`` records that point to DKIM public keys hosted
        by Amazon SES. To complete the DKIM verification process, these tokens
        must be published in the domain's DNS.  The tokens must remain
        published in order for Easy DKIM signing to function correctly.

        After the tokens are added to the domain's DNS, Amazon SES will be able
        to DKIM-sign email originating from that domain.  To enable or disable
        Easy DKIM signing for a domain, use the ``SetIdentityDkimEnabled``
        action.  For more information about Easy DKIM, go to the `Amazon SES
        Developer Guide
        <http://docs.amazonwebservices.com/ses/latest/DeveloperGuide>`_.

        :type domain: string
        :param domain: The domain name.

        """
    return self._make_request('VerifyDomainDkim', {'Domain': domain})
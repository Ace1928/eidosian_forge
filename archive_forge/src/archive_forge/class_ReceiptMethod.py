from keystoneauth1.identity.v3 import base
class ReceiptMethod(base.AuthMethod):
    """Construct an Auth plugin to continue authentication with a receipt.

    :param string receipt: Receipt for authentication.
    """
    _method_parameters = ['receipt']

    def get_auth_data(self, session, auth, headers, **kwargs):
        """Add the auth receipt to the headers.

        We explicitly return None to avoid being added to the request
        methods, or body.
        """
        headers['Openstack-Auth-Receipt'] = self.receipt
        return (None, None)

    def get_cache_id_elements(self):
        return {'receipt_receipt': self.receipt}
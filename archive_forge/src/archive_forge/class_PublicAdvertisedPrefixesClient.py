from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
class PublicAdvertisedPrefixesClient(object):
    """Client for public advertised prefixes service in the GCE API."""

    def __init__(self, client, messages, resources):
        self.client = client
        self.messages = messages
        self.resources = resources
        self._service = self.client.apitools_client.publicAdvertisedPrefixes

    def Create(self, pap_ref, ip_cidr_range, dns_verification_ip, description, pdp_scope):
        """Creates a public advertised prefix."""
        if pdp_scope:
            public_advertised_prefix = self.messages.PublicAdvertisedPrefix(name=pap_ref.Name(), ipCidrRange=ip_cidr_range, dnsVerificationIp=dns_verification_ip, description=description, pdpScope=pdp_scope)
        else:
            public_advertised_prefix = self.messages.PublicAdvertisedPrefix(name=pap_ref.Name(), ipCidrRange=ip_cidr_range, dnsVerificationIp=dns_verification_ip, description=description)
        request = self.messages.ComputePublicAdvertisedPrefixesInsertRequest(publicAdvertisedPrefix=public_advertised_prefix, project=pap_ref.project)
        return self.client.MakeRequests([(self._service, 'Insert', request)])[0]

    def Delete(self, pap_ref):
        """Deletes a public advertised prefix."""
        request = self.messages.ComputePublicAdvertisedPrefixesDeleteRequest(publicAdvertisedPrefix=pap_ref.Name(), project=pap_ref.project)
        return self.client.MakeRequests([(self._service, 'Delete', request)])

    def Announce(self, pap_ref):
        """Announce a public advertised prefix."""
        request = self.messages.ComputePublicAdvertisedPrefixesAnnounceRequest(publicAdvertisedPrefix=pap_ref.Name(), project=pap_ref.project)
        return self.client.MakeRequests([(self._service, 'Announce', request)])

    def Withdraw(self, pap_ref):
        """Withdraw a public advertised prefix."""
        request = self.messages.ComputePublicAdvertisedPrefixesWithdrawRequest(publicAdvertisedPrefix=pap_ref.Name(), project=pap_ref.project)
        return self.client.MakeRequests([(self._service, 'Withdraw', request)])

    def Patch(self, pap_ref, status):
        """Updates public advertised prefix."""
        status = arg_utils.ChoiceToEnum(status, self.messages.PublicAdvertisedPrefix.StatusValueValuesEnum)
        original_pap = self._service.Get(self.client.messages.ComputePublicAdvertisedPrefixesGetRequest(**pap_ref.AsDict()))
        request = self.messages.ComputePublicAdvertisedPrefixesPatchRequest(project=pap_ref.project, publicAdvertisedPrefix=pap_ref.Name(), publicAdvertisedPrefixResource=self.messages.PublicAdvertisedPrefix(status=status, fingerprint=original_pap.fingerprint))
        operation = self._service.Patch(request)
        operation_ref = self.resources.Parse(operation.selfLink, collection='compute.globalOperations')
        operation_poller = poller.Poller(self._service)
        return waiter.WaitFor(operation_poller, operation_ref, 'Updating public advertised prefix [{}].'.format(pap_ref.Name()))
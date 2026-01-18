import abc
import argparse
import os
from stevedore import extension
from troveclient.apiclient import exceptions
@abc.abstractmethod
def token_and_endpoint(self, endpoint_type, service_type):
    """Return token and endpoint.

        :param service_type: Service type of the endpoint
        :type service_type: string
        :param endpoint_type: Type of endpoint.
                              Possible values: public or publicURL,
                              internal or internalURL,
                              admin or adminURL
        :type endpoint_type: string
        :returns: tuple of token and endpoint strings
        :raises: EndpointException
        """
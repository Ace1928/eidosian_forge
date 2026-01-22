from libcloud.common.google import GoogleResponse, GoogleBaseConnection, GoogleOAuth2Credential
from libcloud.container.providers import Provider
from libcloud.container.drivers.kubernetes import KubernetesContainerDriver
class GKEResponse(GoogleResponse):
    pass
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class GrapheneIntegration(Integration):
    identifier = 'graphene'

    @staticmethod
    def setup_once():
        version = package_version('graphene')
        if version is None:
            raise DidNotEnable('Unparsable graphene version.')
        if version < (3, 3):
            raise DidNotEnable('graphene 3.3 or newer required.')
        _patch_graphql()
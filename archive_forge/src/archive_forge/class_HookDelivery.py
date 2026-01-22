from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
class HookDelivery(HookDeliverySummary):
    """
    This class represents a HookDelivery
    """

    def _initAttributes(self) -> None:
        super()._initAttributes()
        self._request: Attribute[HookDeliveryRequest] = NotSet
        self._response: Attribute[HookDeliveryResponse] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def request(self) -> Optional[HookDeliveryRequest]:
        return self._request.value

    @property
    def response(self) -> Optional[HookDeliveryResponse]:
        return self._response.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        super()._useAttributes(attributes)
        if 'request' in attributes:
            self._request = self._makeClassAttribute(HookDeliveryRequest, attributes['request'])
        if 'response' in attributes:
            self._response = self._makeClassAttribute(HookDeliveryResponse, attributes['response'])
from __future__ import annotations
import time
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Action, ActionFailedException, ActionTimeoutException
class BoundAction(BoundModelBase, Action):
    _client: ActionsClient
    model = Action

    def wait_until_finished(self, max_retries: int=100) -> None:
        """Wait until the specific action has status="finished" (set Client.poll_interval to specify a delay between checks)

        :param max_retries: int
               Specify how many retries will be performed before an ActionTimeoutException will be raised
        :raises: ActionFailedException when action is finished with status=="error"
        :raises: ActionTimeoutException when Action is still in "running" state after max_retries reloads.
        """
        while self.status == Action.STATUS_RUNNING:
            if max_retries > 0:
                self.reload()
                time.sleep(self._client._client.poll_interval)
                max_retries = max_retries - 1
            else:
                raise ActionTimeoutException(action=self)
        if self.status == Action.STATUS_ERROR:
            raise ActionFailedException(action=self)
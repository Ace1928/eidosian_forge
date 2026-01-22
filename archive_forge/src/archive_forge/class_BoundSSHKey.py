from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import SSHKey
class BoundSSHKey(BoundModelBase, SSHKey):
    _client: SSHKeysClient
    model = SSHKey

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundSSHKey:
        """Updates an SSH key. You can update an SSH key name and an SSH key labels.

        :param description: str (optional)
               New Description to set
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundSSHKey <hcloud.ssh_keys.client.BoundSSHKey>`
        """
        return self._client.update(self, name, labels)

    def delete(self) -> bool:
        """Deletes an SSH key. It cannot be used anymore.
        :return: boolean
        """
        return self._client.delete(self)
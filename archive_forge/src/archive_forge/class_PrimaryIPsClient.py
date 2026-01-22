from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePrimaryIPResponse, PrimaryIP
class PrimaryIPsClient(ClientEntityBase):
    _client: Client
    actions: ResourceActionsClient
    'Primary IPs scoped actions client\n\n    :type: :class:`ResourceActionsClient <hcloud.actions.client.ResourceActionsClient>`\n    '

    def __init__(self, client: Client):
        super().__init__(client)
        self.actions = ResourceActionsClient(client, '/primary_ips')

    def get_by_id(self, id: int) -> BoundPrimaryIP:
        """Returns a specific Primary IP object.

        :param id: int
        :return: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`
        """
        response = self._client.request(url=f'/primary_ips/{id}', method='GET')
        return BoundPrimaryIP(self, response['primary_ip'])

    def get_list(self, label_selector: str | None=None, page: int | None=None, per_page: int | None=None, name: str | None=None, ip: str | None=None) -> PrimaryIPsPageResult:
        """Get a list of primary ips from this account

        :param label_selector: str (optional)
               Can be used to filter Primary IPs by labels. The response will only contain Primary IPs matching the label selectorable values.
        :param page: int (optional)
               Specifies the page to fetch
        :param per_page: int (optional)
               Specifies how many results are returned by page
        :param name: str (optional)
               Can be used to filter networks by their name.
        :param ip: str (optional)
               Can be used to filter resources by their ip. The response will only contain the resources matching the specified ip.
        :return: (List[:class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`], :class:`Meta <hcloud.core.domain.Meta>`)
        """
        params: dict[str, Any] = {}
        if label_selector is not None:
            params['label_selector'] = label_selector
        if page is not None:
            params['page'] = page
        if per_page is not None:
            params['per_page'] = per_page
        if name is not None:
            params['name'] = name
        if ip is not None:
            params['ip'] = ip
        response = self._client.request(url='/primary_ips', method='GET', params=params)
        primary_ips = [BoundPrimaryIP(self, primary_ip_data) for primary_ip_data in response['primary_ips']]
        return PrimaryIPsPageResult(primary_ips, Meta.parse_meta(response))

    def get_all(self, label_selector: str | None=None, name: str | None=None) -> list[BoundPrimaryIP]:
        """Get all primary ips from this account

        :param label_selector: str (optional)
               Can be used to filter Primary IPs by labels. The response will only contain Primary IPs matching the label selector.able values.
        :param name: str (optional)
               Can be used to filter networks by their name.
        :return: List[:class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`]
        """
        return self._iter_pages(self.get_list, label_selector=label_selector, name=name)

    def get_by_name(self, name: str) -> BoundPrimaryIP | None:
        """Get Primary IP by name

        :param name: str
               Used to get Primary IP by name.
        :return: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`
        """
        return self._get_first_by(name=name)

    def create(self, type: str, datacenter: Datacenter | BoundDatacenter | None, name: str, assignee_type: str | None='server', assignee_id: int | None=None, auto_delete: bool | None=False, labels: dict | None=None) -> CreatePrimaryIPResponse:
        """Creates a new Primary IP assigned to a server.

        :param type: str
               Primary IP type Choices: ipv4, ipv6
        :param assignee_type: str
        :param assignee_id: int (optional)
        :param datacenter: Datacenter
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str
        :param auto_delete: bool (optional)
        :return: :class:`CreatePrimaryIPResponse <hcloud.primary_ips.domain.CreatePrimaryIPResponse>`
        """
        data: dict[str, Any] = {'type': type, 'assignee_type': assignee_type, 'auto_delete': auto_delete, 'name': name}
        if datacenter is not None:
            data['datacenter'] = datacenter.id_or_name
        if assignee_id is not None:
            data['assignee_id'] = assignee_id
        if labels is not None:
            data['labels'] = labels
        response = self._client.request(url='/primary_ips', json=data, method='POST')
        action = None
        if response.get('action') is not None:
            action = BoundAction(self._client.actions, response['action'])
        result = CreatePrimaryIPResponse(primary_ip=BoundPrimaryIP(self, response['primary_ip']), action=action)
        return result

    def update(self, primary_ip: PrimaryIP | BoundPrimaryIP, auto_delete: bool | None=None, labels: dict[str, str] | None=None, name: str | None=None) -> BoundPrimaryIP:
        """Updates the name, auto_delete or labels of a Primary IP.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :param auto_delete: bool (optional)
               Delete this Primary IP when the resource it is assigned to is deleted
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :param name: str (optional)
               New name to set
        :return: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>`
        """
        data: dict[str, Any] = {}
        if auto_delete is not None:
            data['auto_delete'] = auto_delete
        if labels is not None:
            data['labels'] = labels
        if name is not None:
            data['name'] = name
        response = self._client.request(url=f'/primary_ips/{primary_ip.id}', method='PUT', json=data)
        return BoundPrimaryIP(self, response['primary_ip'])

    def delete(self, primary_ip: PrimaryIP | BoundPrimaryIP) -> bool:
        """Deletes a Primary IP. If it is currently assigned to an assignee it will automatically get unassigned.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :return: boolean
        """
        self._client.request(url=f'/primary_ips/{primary_ip.id}', method='DELETE')
        return True

    def change_protection(self, primary_ip: PrimaryIP | BoundPrimaryIP, delete: bool | None=None) -> BoundAction:
        """Changes the protection configuration of the Primary IP.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :param delete: boolean
               If true, prevents the Primary IP from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        data: dict[str, Any] = {}
        if delete is not None:
            data.update({'delete': delete})
        response = self._client.request(url=f'/primary_ips/{primary_ip.id}/actions/change_protection', method='POST', json=data)
        return BoundAction(self._client.actions, response['action'])

    def assign(self, primary_ip: PrimaryIP | BoundPrimaryIP, assignee_id: int, assignee_type: str='server') -> BoundAction:
        """Assigns a Primary IP to a assignee_id.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :param assignee_id: int
               Assignee the Primary IP shall be assigned to
        :param assignee_type: str
               Assignee the Primary IP shall be assigned to
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        response = self._client.request(url=f'/primary_ips/{primary_ip.id}/actions/assign', method='POST', json={'assignee_id': assignee_id, 'assignee_type': assignee_type})
        return BoundAction(self._client.actions, response['action'])

    def unassign(self, primary_ip: PrimaryIP | BoundPrimaryIP) -> BoundAction:
        """Unassigns a Primary IP, resulting in it being unreachable. You may assign it to a server again at a later time.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        response = self._client.request(url=f'/primary_ips/{primary_ip.id}/actions/unassign', method='POST')
        return BoundAction(self._client.actions, response['action'])

    def change_dns_ptr(self, primary_ip: PrimaryIP | BoundPrimaryIP, ip: str, dns_ptr: str) -> BoundAction:
        """Changes the dns ptr that will appear when getting the dns ptr belonging to this Primary IP.

        :param primary_ip: :class:`BoundPrimaryIP <hcloud.primary_ips.client.BoundPrimaryIP>` or  :class:`PrimaryIP <hcloud.primary_ips.domain.PrimaryIP>`
        :param ip: str
               The IP address for which to set the reverse DNS entry
        :param dns_ptr: str
               Hostname to set as a reverse DNS PTR entry, will reset to original default value if `None`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        response = self._client.request(url=f'/primary_ips/{primary_ip.id}/actions/change_dns_ptr', method='POST', json={'ip': ip, 'dns_ptr': dns_ptr})
        return BoundAction(self._client.actions, response['action'])
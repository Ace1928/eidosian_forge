from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..certificates import BoundCertificate
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..load_balancer_types import BoundLoadBalancerType
from ..locations import BoundLocation
from ..metrics import Metrics
from ..networks import BoundNetwork
from ..servers import BoundServer
from .domain import (
class BoundLoadBalancer(BoundModelBase, LoadBalancer):
    _client: LoadBalancersClient
    model = LoadBalancer

    def __init__(self, client: LoadBalancersClient, data: dict, complete: bool=True):
        algorithm = data.get('algorithm')
        if algorithm:
            data['algorithm'] = LoadBalancerAlgorithm(type=algorithm['type'])
        public_net = data.get('public_net')
        if public_net:
            ipv4_address = IPv4Address.from_dict(public_net['ipv4'])
            ipv6_network = IPv6Network.from_dict(public_net['ipv6'])
            data['public_net'] = PublicNetwork(ipv4=ipv4_address, ipv6=ipv6_network, enabled=public_net['enabled'])
        private_nets = data.get('private_net')
        if private_nets:
            private_nets = [PrivateNet(network=BoundNetwork(client._client.networks, {'id': private_net['network']}, complete=False), ip=private_net['ip']) for private_net in private_nets]
            data['private_net'] = private_nets
        targets = data.get('targets')
        if targets:
            tmp_targets = []
            for target in targets:
                tmp_target = LoadBalancerTarget(type=target['type'])
                if target['type'] == 'server':
                    tmp_target.server = BoundServer(client._client.servers, data=target['server'], complete=False)
                    tmp_target.use_private_ip = target['use_private_ip']
                elif target['type'] == 'label_selector':
                    tmp_target.label_selector = LoadBalancerTargetLabelSelector(selector=target['label_selector']['selector'])
                    tmp_target.use_private_ip = target['use_private_ip']
                elif target['type'] == 'ip':
                    tmp_target.ip = LoadBalancerTargetIP(ip=target['ip']['ip'])
                target_health_status = target.get('health_status')
                if target_health_status is not None:
                    tmp_target.health_status = [LoadBalancerTargetHealthStatus(listen_port=target_health_status_item['listen_port'], status=target_health_status_item['status']) for target_health_status_item in target_health_status]
                tmp_targets.append(tmp_target)
            data['targets'] = tmp_targets
        services = data.get('services')
        if services:
            tmp_services = []
            for service in services:
                tmp_service = LoadBalancerService(protocol=service['protocol'], listen_port=service['listen_port'], destination_port=service['destination_port'], proxyprotocol=service['proxyprotocol'])
                if service['protocol'] != 'tcp':
                    tmp_service.http = LoadBalancerServiceHttp(sticky_sessions=service['http']['sticky_sessions'], redirect_http=service['http']['redirect_http'], cookie_name=service['http']['cookie_name'], cookie_lifetime=service['http']['cookie_lifetime'])
                    tmp_service.http.certificates = [BoundCertificate(client._client.certificates, {'id': certificate}, complete=False) for certificate in service['http']['certificates']]
                tmp_service.health_check = LoadBalancerHealthCheck(protocol=service['health_check']['protocol'], port=service['health_check']['port'], interval=service['health_check']['interval'], retries=service['health_check']['retries'], timeout=service['health_check']['timeout'])
                if tmp_service.health_check.protocol != 'tcp':
                    tmp_service.health_check.http = LoadBalancerHealtCheckHttp(domain=service['health_check']['http']['domain'], path=service['health_check']['http']['path'], response=service['health_check']['http']['response'], tls=service['health_check']['http']['tls'], status_codes=service['health_check']['http']['status_codes'])
                tmp_services.append(tmp_service)
            data['services'] = tmp_services
        load_balancer_type = data.get('load_balancer_type')
        if load_balancer_type is not None:
            data['load_balancer_type'] = BoundLoadBalancerType(client._client.load_balancer_types, load_balancer_type)
        location = data.get('location')
        if location is not None:
            data['location'] = BoundLocation(client._client.locations, location)
        super().__init__(client, data, complete)

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundLoadBalancer:
        """Updates a Load Balancer. You can update a Load Balancers name and a Load Balancers labels.

        :param name: str (optional)
               New name to set
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundLoadBalancer <hcloud.load_balancers.client.BoundLoadBalancer>`
        """
        return self._client.update(self, name, labels)

    def delete(self) -> bool:
        """Deletes a Load Balancer.

        :return: boolean
        """
        return self._client.delete(self)

    def get_metrics(self, type: MetricsType, start: datetime | str, end: datetime | str, step: float | None=None) -> GetMetricsResponse:
        """Get Metrics for a LoadBalancer.

        :param type: Type of metrics to get.
        :param start: Start of period to get Metrics for (in ISO-8601 format).
        :param end: End of period to get Metrics for (in ISO-8601 format).
        :param step: Resolution of results in seconds.
        """
        return self._client.get_metrics(self, type=type, start=start, end=end, step=step)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a Load Balancer.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :param page: int (optional)
               Specifies the page to fetch
        :param per_page: int (optional)
               Specifies how many results are returned by page
        :return: (List[:class:`BoundAction <hcloud.actions.client.BoundAction>`], :class:`Meta <hcloud.core.domain.Meta>`)
        """
        return self._client.get_actions_list(self, status, sort, page, per_page)

    def get_actions(self, status: list[str] | None=None, sort: list[str] | None=None) -> list[BoundAction]:
        """Returns all action objects for a Load Balancer.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def add_service(self, service: LoadBalancerService) -> BoundAction:
        """Adds a service to a Load Balancer.

        :param service: :class:`LoadBalancerService <hcloud.load_balancers.domain.LoadBalancerService>`
                       The LoadBalancerService you want to add to the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.add_service(self, service=service)

    def update_service(self, service: LoadBalancerService) -> BoundAction:
        """Updates a service of an Load Balancer.

        :param service: :class:`LoadBalancerService <hcloud.load_balancers.domain.LoadBalancerService>`
                       The LoadBalancerService you  want to update
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.update_service(self, service=service)

    def delete_service(self, service: LoadBalancerService) -> BoundAction:
        """Deletes a service from a Load Balancer.

        :param service: :class:`LoadBalancerService <hcloud.load_balancers.domain.LoadBalancerService>`
                       The LoadBalancerService you want to delete from the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.delete_service(self, service)

    def add_target(self, target: LoadBalancerTarget) -> BoundAction:
        """Adds a target to a Load Balancer.

        :param target: :class:`LoadBalancerTarget <hcloud.load_balancers.domain.LoadBalancerTarget>`
                       The LoadBalancerTarget you want to add to the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.add_target(self, target)

    def remove_target(self, target: LoadBalancerTarget) -> BoundAction:
        """Removes a target from a Load Balancer.

        :param target: :class:`LoadBalancerTarget <hcloud.load_balancers.domain.LoadBalancerTarget>`
                       The LoadBalancerTarget you want to remove from the Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.remove_target(self, target)

    def change_algorithm(self, algorithm: LoadBalancerAlgorithm) -> BoundAction:
        """Changes the algorithm used by the Load Balancer

        :param algorithm: :class:`LoadBalancerAlgorithm <hcloud.load_balancers.domain.LoadBalancerAlgorithm>`
                       The LoadBalancerAlgorithm you want to use
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_algorithm(self, algorithm)

    def change_dns_ptr(self, ip: str, dns_ptr: str) -> BoundAction:
        """Changes the hostname that will appear when getting the hostname belonging to the public IPs (IPv4 and IPv6) of this Load Balancer.

        :param ip: str
               The IP address for which to set the reverse DNS entry
        :param dns_ptr: str
               Hostname to set as a reverse DNS PTR entry, will reset to original default value if `None`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_dns_ptr(self, ip, dns_ptr)

    def change_protection(self, delete: bool) -> BoundAction:
        """Changes the protection configuration of a Load Balancer.

        :param delete: boolean
               If True, prevents the Load Balancer from being deleted
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_protection(self, delete)

    def attach_to_network(self, network: Network | BoundNetwork, ip: str | None=None) -> BoundAction:
        """Attaches a Load Balancer to a Network

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param ip: str
                IP to request to be assigned to this Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.attach_to_network(self, network, ip)

    def detach_from_network(self, network: Network | BoundNetwork) -> BoundAction:
        """Detaches a Load Balancer from a Network.

        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.detach_from_network(self, network)

    def enable_public_interface(self) -> BoundAction:
        """Enables the public interface of a Load Balancer.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.enable_public_interface(self)

    def disable_public_interface(self) -> BoundAction:
        """Disables the public interface of a Load Balancer.

        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.disable_public_interface(self)

    def change_type(self, load_balancer_type: LoadBalancerType | BoundLoadBalancerType) -> BoundAction:
        """Changes the type of a Load Balancer.

        :param load_balancer_type: :class:`BoundLoadBalancerType <hcloud.load_balancer_types.client.BoundLoadBalancerType>` or :class:`LoadBalancerType <hcloud.load_balancer_types.domain.LoadBalancerType>`
               Load Balancer type the Load Balancer should migrate to
        :return:  :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
        return self._client.change_type(self, load_balancer_type)
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
class RackspaceLBDriver(Driver, OpenStackDriverMixin):
    connectionCls = RackspaceConnection
    api_name = 'rackspace_lb'
    name = 'Rackspace LB'
    website = 'http://www.rackspace.com/'
    LB_STATE_MAP = {'ACTIVE': State.RUNNING, 'BUILD': State.PENDING, 'ERROR': State.ERROR, 'DELETED': State.DELETED, 'PENDING_UPDATE': State.PENDING, 'PENDING_DELETE': State.PENDING}
    LB_MEMBER_CONDITION_MAP = {'ENABLED': MemberCondition.ENABLED, 'DISABLED': MemberCondition.DISABLED, 'DRAINING': MemberCondition.DRAINING}
    CONDITION_LB_MEMBER_MAP = reverse_dict(LB_MEMBER_CONDITION_MAP)
    _VALUE_TO_ALGORITHM_MAP = {'RANDOM': Algorithm.RANDOM, 'ROUND_ROBIN': Algorithm.ROUND_ROBIN, 'LEAST_CONNECTIONS': Algorithm.LEAST_CONNECTIONS, 'WEIGHTED_ROUND_ROBIN': Algorithm.WEIGHTED_ROUND_ROBIN, 'WEIGHTED_LEAST_CONNECTIONS': Algorithm.WEIGHTED_LEAST_CONNECTIONS}
    _ALGORITHM_TO_VALUE_MAP = reverse_dict(_VALUE_TO_ALGORITHM_MAP)

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='ord', **kwargs):
        ex_force_region = kwargs.pop('ex_force_region', None)
        if ex_force_region:
            region = ex_force_region
        OpenStackDriverMixin.__init__(self, **kwargs)
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region)

    @classmethod
    def list_regions(cls):
        return ENDPOINT_ARGS_MAP.keys()

    def _ex_connection_class_kwargs(self):
        endpoint_args = ENDPOINT_ARGS_MAP[self.region]
        kwargs = self.openstack_connection_kwargs()
        kwargs['get_endpoint_args'] = endpoint_args
        return kwargs

    def list_protocols(self):
        return self._to_protocols(self.connection.request('/loadbalancers/protocols').object)

    def ex_list_protocols_with_default_ports(self):
        """
        List protocols with default ports.

        :rtype: ``list`` of ``tuple``
        :return: A list of protocols with default ports included.
        """
        return self._to_protocols_with_default_ports(self.connection.request('/loadbalancers/protocols').object)

    def list_balancers(self, ex_member_address=None, ex_status=None, ex_changes_since=None, ex_params={}):
        """
        @inherits: :class:`Driver.list_balancers`

        :param ex_member_address: Optional IP address of the attachment member.
                                  If provided, only the load balancers which
                                  have this member attached will be returned.
        :type ex_member_address: ``str``

        :param ex_status: Optional. Filter balancers by status
        :type ex_status: ``str``

        :param ex_changes_since: Optional. List all load balancers that have
                                 changed since the specified date/time
        :type ex_changes_since: ``str``

        :param ex_params: Optional. Set parameters to be submitted to the API
                          in the query string
        :type ex_params: ``dict``
        """
        params = {}
        if ex_member_address:
            params['nodeaddress'] = ex_member_address
        if ex_status:
            params['status'] = ex_status
        if ex_changes_since:
            params['changes-since'] = ex_changes_since
        for key, value in ex_params.items():
            params[key] = value
        return self._to_balancers(self.connection.request('/loadbalancers', params=params).object)

    def create_balancer(self, name, members, protocol='http', port=80, algorithm=DEFAULT_ALGORITHM):
        return self.ex_create_balancer(name, members, protocol, port, algorithm)

    def ex_create_balancer(self, name, members, protocol='http', port=80, algorithm=DEFAULT_ALGORITHM, vip='PUBLIC'):
        """
        Creates a new load balancer instance

        :param name: Name of the new load balancer (required)
        :type  name: ``str``

        :param members: ``list`` of:class:`Member`s to attach to balancer
        :type  members: ``list`` of :class:`Member`

        :param protocol: Loadbalancer protocol, defaults to http.
        :type  protocol: ``str``

        :param port: Port the load balancer should listen on, defaults to 80
        :type  port: ``str``

        :param algorithm: Load balancing algorithm, defaults to
                            LBAlgorithm.ROUND_ROBIN
        :type  algorithm: :class:`Algorithm`

        :param vip: Virtual ip type of PUBLIC, SERVICENET, or ID of a virtual
                      ip
        :type  vip: ``str``

        :rtype: :class:`LoadBalancer`
        """
        balancer_attrs = self._kwargs_to_mutable_attrs(name=name, protocol=protocol, port=port, algorithm=algorithm, vip=vip)
        balancer_attrs.update({'nodes': [self._member_attributes(member) for member in members]})
        balancer_object = {'loadBalancer': balancer_attrs}
        resp = self.connection.request('/loadbalancers', method='POST', data=json.dumps(balancer_object))
        return self._to_balancer(resp.object['loadBalancer'])

    def _member_attributes(self, member):
        member_attributes = {'address': member.ip, 'port': member.port}
        member_attributes.update(self._kwargs_to_mutable_member_attrs(**member.extra))
        if 'condition' not in member_attributes:
            member_attributes['condition'] = self.CONDITION_LB_MEMBER_MAP[MemberCondition.ENABLED]
        return member_attributes

    def destroy_balancer(self, balancer):
        uri = '/loadbalancers/%s' % balancer.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_destroy_balancers(self, balancers):
        """
        Destroys a list of Balancers (the API supports up to 10).

        :param balancers: A list of Balancers to destroy.
        :type balancers: ``list`` of :class:`LoadBalancer`

        :return: Returns whether the destroy request was accepted.
        :rtype: ``bool``
        """
        ids = [('id', balancer.id) for balancer in balancers]
        resp = self.connection.request('/loadbalancers', method='DELETE', params=ids)
        return resp.status == httplib.ACCEPTED

    def get_balancer(self, balancer_id):
        uri = '/loadbalancers/%s' % balancer_id
        resp = self.connection.request(uri)
        return self._to_balancer(resp.object['loadBalancer'])

    def balancer_attach_member(self, balancer, member):
        member_object = {'nodes': [self._member_attributes(member)]}
        uri = '/loadbalancers/%s/nodes' % balancer.id
        resp = self.connection.request(uri, method='POST', data=json.dumps(member_object))
        return self._to_members(resp.object, balancer)[0]

    def ex_balancer_attach_members(self, balancer, members):
        """
        Attaches a list of members to a load balancer.

        :param balancer: The Balancer to which members will be attached.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to attach.
        :type  members: ``list`` of :class:`Member`

        :rtype: ``list`` of :class:`Member`
        """
        member_objects = {'nodes': [self._member_attributes(member) for member in members]}
        uri = '/loadbalancers/%s/nodes' % balancer.id
        resp = self.connection.request(uri, method='POST', data=json.dumps(member_objects))
        return self._to_members(resp.object, balancer)

    def balancer_detach_member(self, balancer, member):
        uri = '/loadbalancers/{}/nodes/{}'.format(balancer.id, member.id)
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_balancer_detach_members(self, balancer, members):
        """
        Detaches a list of members from a balancer (the API supports up to 10).
        This method blocks until the detach request has been processed and the
        balancer is in a RUNNING state again.

        :param balancer: The Balancer to detach members from.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to detach.
        :type  members: ``list`` of :class:`Member`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        accepted = self.ex_balancer_detach_members_no_poll(balancer, members)
        if not accepted:
            msg = 'Detach members request was not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_balancer_detach_members_no_poll(self, balancer, members):
        """
        Detaches a list of members from a balancer (the API supports up to 10).
        This method returns immediately.

        :param balancer: The Balancer to detach members from.
        :type  balancer: :class:`LoadBalancer`

        :param members: A list of Members to detach.
        :type  members: ``list`` of :class:`Member`

        :return: Returns whether the detach request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/nodes' % balancer.id
        ids = [('id', member.id) for member in members]
        resp = self.connection.request(uri, method='DELETE', params=ids)
        return resp.status == httplib.ACCEPTED

    def balancer_list_members(self, balancer):
        uri = '/loadbalancers/%s/nodes' % balancer.id
        data = self.connection.request(uri).object
        return self._to_members(data, balancer)

    def update_balancer(self, balancer, **kwargs):
        attrs = self._kwargs_to_mutable_attrs(**kwargs)
        resp = self.connection.async_request(action='/loadbalancers/%s' % balancer.id, method='PUT', data=json.dumps(attrs))
        return self._to_balancer(resp.object['loadBalancer'])

    def ex_update_balancer_no_poll(self, balancer, **kwargs):
        """
        Update balancer no poll.

        @inherits: :class:`Driver.update_balancer`
        """
        attrs = self._kwargs_to_mutable_attrs(**kwargs)
        resp = self.connection.request(action='/loadbalancers/%s' % balancer.id, method='PUT', data=json.dumps(attrs))
        return resp.status == httplib.ACCEPTED

    def ex_balancer_update_member(self, balancer, member, **kwargs):
        """
        Updates a Member's extra attributes for a Balancer.  The attributes can
        include 'weight' or 'condition'.  This method blocks until the update
        request has been processed and the balancer is in a RUNNING state
        again.

        :param balancer: Balancer to update the member on.
        :type  balancer: :class:`LoadBalancer`

        :param member: Member which should be used
        :type member: :class:`Member`

        :keyword **kwargs: New attributes.  Should contain either 'weight'
        or 'condition'.  'condition' can be set to 'ENABLED', 'DISABLED'.
        or 'DRAINING'.  'weight' can be set to a positive integer between
        1 and 100, with a higher weight indicating that the node will receive
        more traffic (assuming the Balancer is using a weighted algorithm).
        :type **kwargs: ``dict``

        :return: Updated Member.
        :rtype: :class:`Member`
        """
        accepted = self.ex_balancer_update_member_no_poll(balancer, member, **kwargs)
        if not accepted:
            msg = 'Update member attributes was not accepted'
            raise LibcloudError(msg, driver=self)
        balancer = self._get_updated_balancer(balancer)
        members = balancer.extra['members']
        updated_members = [m for m in members if m.id == member.id]
        if not updated_members:
            raise LibcloudError('Could not find updated member')
        return updated_members[0]

    def ex_balancer_update_member_no_poll(self, balancer, member, **kwargs):
        """
        Updates a Member's extra attributes for a Balancer.  The attribute can
        include 'weight' or 'condition'.  This method returns immediately.

        :param balancer: Balancer to update the member on.
        :type balancer: :class:`LoadBalancer`

        :param member: Member which should be used
        :type member: :class:`Member`

        :keyword **kwargs: New attributes.  Should contain either 'weight'
        or 'condition'.  'condition' can be set to 'ENABLED', 'DISABLED'.
        or 'DRAINING'.  'weight' can be set to a positive integer between
        1 and 100, with a higher weight indicating that the node will receive
        more traffic (assuming the Balancer is using a weighted algorithm).
        :type **kwargs: ``dict``

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
        resp = self.connection.request(action='/loadbalancers/{}/nodes/{}'.format(balancer.id, member.id), method='PUT', data=json.dumps(self._kwargs_to_mutable_member_attrs(**kwargs)))
        return resp.status == httplib.ACCEPTED

    def ex_list_algorithm_names(self):
        """
        Lists algorithms supported by the API.  Returned as strings because
        this list may change in the future.

        :rtype: ``list`` of ``str``
        """
        response = self.connection.request('/loadbalancers/algorithms')
        return [a['name'].upper() for a in response.object['algorithms']]

    def ex_get_balancer_error_page(self, balancer):
        """
        List error page configured for the specified load balancer.

        :param balancer: Balancer which should be used
        :type balancer: :class:`LoadBalancer`

        :rtype: ``str``
        """
        uri = '/loadbalancers/%s/errorpage' % balancer.id
        resp = self.connection.request(uri)
        return resp.object['errorpage']['content']

    def ex_balancer_access_list(self, balancer):
        """
        List the access list.

        :param balancer: Balancer which should be used
        :type balancer: :class:`LoadBalancer`

        :rtype: ``list`` of :class:`RackspaceAccessRule`
        """
        uri = '/loadbalancers/%s/accesslist' % balancer.id
        resp = self.connection.request(uri)
        return [self._to_access_rule(el) for el in resp.object['accessList']]

    def _get_updated_balancer(self, balancer):
        """
        Updating a balancer's attributes puts a balancer into
        'PENDING_UPDATE' status.  Wait until the balancer is
        back in 'ACTIVE' status and then return the individual
        balancer details call.
        """
        resp = self.connection.async_request(action='/loadbalancers/%s' % balancer.id, method='GET')
        return self._to_balancer(resp.object['loadBalancer'])

    def ex_update_balancer_health_monitor(self, balancer, health_monitor):
        """
        Sets a Balancer's health monitor.  This method blocks until the update
        request has been processed and the balancer is in a RUNNING state
        again.

        :param balancer: Balancer to update.
        :type  balancer: :class:`LoadBalancer`

        :param health_monitor: Health Monitor for the balancer.
        :type  health_monitor: :class:`RackspaceHealthMonitor`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        accepted = self.ex_update_balancer_health_monitor_no_poll(balancer, health_monitor)
        if not accepted:
            msg = 'Update health monitor request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_update_balancer_health_monitor_no_poll(self, balancer, health_monitor):
        """
        Sets a Balancer's health monitor.  This method returns immediately.

        :param balancer: Balancer to update health monitor on.
        :type  balancer: :class:`LoadBalancer`

        :param health_monitor: Health Monitor for the balancer.
        :type  health_monitor: :class:`RackspaceHealthMonitor`

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/healthmonitor' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps(health_monitor._to_dict()))
        return resp.status == httplib.ACCEPTED

    def ex_disable_balancer_health_monitor(self, balancer):
        """
        Disables a Balancer's health monitor.  This method blocks until the
        disable request has been processed and the balancer is in a RUNNING
        state again.

        :param balancer: Balancer to disable health monitor on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_disable_balancer_health_monitor_no_poll(balancer):
            msg = 'Disable health monitor request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_disable_balancer_health_monitor_no_poll(self, balancer):
        """
        Disables a Balancer's health monitor.  This method returns
        immediately.

        :param balancer: Balancer to disable health monitor on.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/healthmonitor' % balancer.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_update_balancer_connection_throttle(self, balancer, connection_throttle):
        """
        Updates a Balancer's connection throttle.  This method blocks until
        the update request has been processed and the balancer is in a
        RUNNING state again.

        :param balancer: Balancer to update connection throttle on.
        :type  balancer: :class:`LoadBalancer`

        :param connection_throttle: Connection Throttle for the balancer.
        :type  connection_throttle: :class:`RackspaceConnectionThrottle`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        accepted = self.ex_update_balancer_connection_throttle_no_poll(balancer, connection_throttle)
        if not accepted:
            msg = 'Update connection throttle request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_update_balancer_connection_throttle_no_poll(self, balancer, connection_throttle):
        """
        Sets a Balancer's connection throttle.  This method returns
        immediately.

        :param balancer: Balancer to update connection throttle on.
        :type  balancer: :class:`LoadBalancer`

        :param connection_throttle: Connection Throttle for the balancer.
        :type  connection_throttle: :class:`RackspaceConnectionThrottle`

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/connectionthrottle' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps(connection_throttle._to_dict()))
        return resp.status == httplib.ACCEPTED

    def ex_disable_balancer_connection_throttle(self, balancer):
        """
        Disables a Balancer's connection throttle.  This method blocks until
        the disable request has been processed and the balancer is in a RUNNING
        state again.

        :param balancer: Balancer to disable connection throttle on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_disable_balancer_connection_throttle_no_poll(balancer):
            msg = 'Disable connection throttle request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_disable_balancer_connection_throttle_no_poll(self, balancer):
        """
        Disables a Balancer's connection throttle.  This method returns
        immediately.

        :param balancer: Balancer to disable connection throttle on.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/connectionthrottle' % balancer.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_enable_balancer_connection_logging(self, balancer):
        """
        Enables connection logging for a Balancer.  This method blocks until
        the enable request has been processed and the balancer is in a RUNNING
        state again.

        :param balancer: Balancer to enable connection logging on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_enable_balancer_connection_logging_no_poll(balancer):
            msg = 'Enable connection logging request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_enable_balancer_connection_logging_no_poll(self, balancer):
        """
        Enables connection logging for a Balancer.  This method returns
        immediately.

        :param balancer: Balancer to enable connection logging on.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the enable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/connectionlogging' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps({'connectionLogging': {'enabled': True}}))
        return resp.status == httplib.ACCEPTED

    def ex_disable_balancer_connection_logging(self, balancer):
        """
        Disables connection logging for a Balancer.  This method blocks until
        the enable request has been processed and the balancer is in a RUNNING
        state again.

        :param balancer: Balancer to disable connection logging on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_disable_balancer_connection_logging_no_poll(balancer):
            msg = 'Disable connection logging request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_disable_balancer_connection_logging_no_poll(self, balancer):
        """
        Disables connection logging for a Balancer.  This method returns
        immediately.

        :param balancer: Balancer to disable connection logging on.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/connectionlogging' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps({'connectionLogging': {'enabled': False}}))
        return resp.status == httplib.ACCEPTED

    def ex_enable_balancer_session_persistence(self, balancer):
        """
        Enables session persistence for a Balancer by setting the persistence
        type to 'HTTP_COOKIE'.  This method blocks until the enable request
        has been processed and the balancer is in a RUNNING state again.

        :param balancer: Balancer to enable session persistence on.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_enable_balancer_session_persistence_no_poll(balancer):
            msg = 'Enable session persistence request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_enable_balancer_session_persistence_no_poll(self, balancer):
        """
        Enables session persistence for a Balancer by setting the persistence
        type to 'HTTP_COOKIE'.  This method returns immediately.

        :param balancer: Balancer to enable session persistence on.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the enable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/sessionpersistence' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps({'sessionPersistence': {'persistenceType': 'HTTP_COOKIE'}}))
        return resp.status == httplib.ACCEPTED

    def ex_disable_balancer_session_persistence(self, balancer):
        """
        Disables session persistence for a Balancer.  This method blocks until
        the disable request has been processed and the balancer is in a RUNNING
        state again.

        :param balancer: Balancer to disable session persistence on.
        :type balancer:  :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_disable_balancer_session_persistence_no_poll(balancer):
            msg = 'Disable session persistence request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_disable_balancer_session_persistence_no_poll(self, balancer):
        """
        Disables session persistence for a Balancer.  This method returns
        immediately.

        :param balancer: Balancer to disable session persistence for.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/sessionpersistence' % balancer.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_update_balancer_error_page(self, balancer, page_content):
        """
        Updates a Balancer's custom error page.  This method blocks until
        the update request has been processed and the balancer is in a
        RUNNING state again.

        :param balancer: Balancer to update the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :param page_content: HTML content for the custom error page.
        :type  page_content: ``str``

        :return: Updated Balancer.
        :rtype:  :class:`LoadBalancer`
        """
        accepted = self.ex_update_balancer_error_page_no_poll(balancer, page_content)
        if not accepted:
            msg = 'Update error page request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_update_balancer_error_page_no_poll(self, balancer, page_content):
        """
        Updates a Balancer's custom error page.  This method returns
        immediately.

        :param balancer: Balancer to update the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :param page_content: HTML content for the custom error page.
        :type  page_content: ``str``

        :return: Returns whether the update request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/errorpage' % balancer.id
        resp = self.connection.request(uri, method='PUT', data=json.dumps({'errorpage': {'content': page_content}}))
        return resp.status == httplib.ACCEPTED

    def ex_disable_balancer_custom_error_page(self, balancer):
        """
        Disables a Balancer's custom error page, returning its error page to
        the Rackspace-provided default.  This method blocks until the disable
        request has been processed and the balancer is in a RUNNING state
        again.

        :param balancer: Balancer to disable the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        if not self.ex_disable_balancer_custom_error_page_no_poll(balancer):
            msg = 'Disable custom error page request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_disable_balancer_custom_error_page_no_poll(self, balancer):
        """
        Disables a Balancer's custom error page, returning its error page to
        the Rackspace-provided default.  This method returns immediately.

        :param balancer: Balancer to disable the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/errorpage' % balancer.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.OK or resp.status == httplib.ACCEPTED

    def ex_create_balancer_access_rule(self, balancer, rule):
        """
        Adds an access rule to a Balancer's access list.  This method blocks
        until the update request has been processed and the balancer is in a
        RUNNING state again.

        :param balancer: Balancer to create the access rule for.
        :type balancer: :class:`LoadBalancer`

        :param rule: Access Rule to add to the balancer.
        :type rule: :class:`RackspaceAccessRule`

        :return: The created access rule.
        :rtype: :class:`RackspaceAccessRule`
        """
        accepted = self.ex_create_balancer_access_rule_no_poll(balancer, rule)
        if not accepted:
            msg = 'Create access rule not accepted'
            raise LibcloudError(msg, driver=self)
        balancer = self._get_updated_balancer(balancer)
        access_list = balancer.extra['accessList']
        created_rule = self._find_matching_rule(rule, access_list)
        if not created_rule:
            raise LibcloudError('Could not find created rule')
        return created_rule

    def ex_create_balancer_access_rule_no_poll(self, balancer, rule):
        """
        Adds an access rule to a Balancer's access list.  This method returns
        immediately.

        :param balancer: Balancer to create the access rule for.
        :type balancer: :class:`LoadBalancer`

        :param rule: Access Rule to add to the balancer.
        :type rule: :class:`RackspaceAccessRule`

        :return: Returns whether the create request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/accesslist' % balancer.id
        resp = self.connection.request(uri, method='POST', data=json.dumps({'networkItem': rule._to_dict()}))
        return resp.status == httplib.ACCEPTED

    def ex_create_balancer_access_rules(self, balancer, rules):
        """
        Adds a list of access rules to a Balancer's access list.  This method
        blocks until the update request has been processed and the balancer is
        in a RUNNING state again.

        :param balancer: Balancer to create the access rule for.
        :type  balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` to add to the
                      balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: The created access rules.
        :rtype: :class:`RackspaceAccessRule`
        """
        accepted = self.ex_create_balancer_access_rules_no_poll(balancer, rules)
        if not accepted:
            msg = 'Create access rules not accepted'
            raise LibcloudError(msg, driver=self)
        balancer = self._get_updated_balancer(balancer)
        access_list = balancer.extra['accessList']
        created_rules = []
        for r in rules:
            matched_rule = self._find_matching_rule(r, access_list)
            if matched_rule:
                created_rules.append(matched_rule)
        if len(created_rules) != len(rules):
            raise LibcloudError('Could not find all created rules')
        return created_rules

    def _find_matching_rule(self, rule_to_find, access_list):
        """
        LB API does not return the ID for the newly created rules, so we have
        to search the list to find the rule with a matching rule type and
        address to return an object with the right identifier.it.  The API
        enforces rule type and address uniqueness.
        """
        for r in access_list:
            if rule_to_find.rule_type == r.rule_type and rule_to_find.address == r.address:
                return r
        return None

    def ex_create_balancer_access_rules_no_poll(self, balancer, rules):
        """
        Adds a list of access rules to a Balancer's access list.  This method
        returns immediately.

        :param balancer: Balancer to create the access rule for.
        :type balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` to add to
                      the balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: Returns whether the create request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/%s/accesslist' % balancer.id
        resp = self.connection.request(uri, method='POST', data=json.dumps({'accessList': [rule._to_dict() for rule in rules]}))
        return resp.status == httplib.ACCEPTED

    def ex_destroy_balancer_access_rule(self, balancer, rule):
        """
        Removes an access rule from a Balancer's access list.  This method
        blocks until the update request has been processed and the balancer
        is in a RUNNING state again.

        :param balancer: Balancer to remove the access rule from.
        :type  balancer: :class:`LoadBalancer`

        :param rule: Access Rule to remove from the balancer.
        :type  rule: :class:`RackspaceAccessRule`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        accepted = self.ex_destroy_balancer_access_rule_no_poll(balancer, rule)
        if not accepted:
            msg = 'Delete access rule not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_destroy_balancer_access_rule_no_poll(self, balancer, rule):
        """
        Removes an access rule from a Balancer's access list.  This method
        returns immediately.

        :param balancer: Balancer to remove the access rule from.
        :type  balancer: :class:`LoadBalancer`

        :param rule: Access Rule to remove from the balancer.
        :type  rule: :class:`RackspaceAccessRule`

        :return: Returns whether the destroy request was accepted.
        :rtype: ``bool``
        """
        uri = '/loadbalancers/{}/accesslist/{}'.format(balancer.id, rule.id)
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_destroy_balancer_access_rules(self, balancer, rules):
        """
        Removes a list of access rules from a Balancer's access list.  This
        method blocks until the update request has been processed and the
        balancer is in a RUNNING state again.

        :param balancer: Balancer to remove the access rules from.
        :type  balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` objects to remove
                      from the balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: Updated Balancer.
        :rtype: :class:`LoadBalancer`
        """
        accepted = self.ex_destroy_balancer_access_rules_no_poll(balancer, rules)
        if not accepted:
            msg = 'Destroy access rules request not accepted'
            raise LibcloudError(msg, driver=self)
        return self._get_updated_balancer(balancer)

    def ex_destroy_balancer_access_rules_no_poll(self, balancer, rules):
        """
        Removes a list of access rules from a Balancer's access list.  This
        method returns immediately.

        :param balancer: Balancer to remove the access rules from.
        :type  balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` objects to remove
                      from the balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: Returns whether the destroy request was accepted.
        :rtype: ``bool``
        """
        ids = [('id', rule.id) for rule in rules]
        uri = '/loadbalancers/%s/accesslist' % balancer.id
        resp = self.connection.request(uri, method='DELETE', params=ids)
        return resp.status == httplib.ACCEPTED

    def ex_list_current_usage(self, balancer):
        """
        Return current load balancer usage report.

        :param balancer: Balancer to remove the access rules from.
        :type  balancer: :class:`LoadBalancer`

        :return: Raw load balancer usage object.
        :rtype: ``dict``
        """
        uri = '/loadbalancers/%s/usage/current' % balancer.id
        resp = self.connection.request(uri, method='GET')
        return resp.object

    def _to_protocols(self, object):
        protocols = []
        for item in object['protocols']:
            protocols.append(item['name'].lower())
        return protocols

    def _to_protocols_with_default_ports(self, object):
        protocols = []
        for item in object['protocols']:
            name = item['name'].lower()
            port = int(item['port'])
            protocols.append((name, port))
        return protocols

    def _to_balancers(self, object):
        return [self._to_balancer(el) for el in object['loadBalancers']]

    def _to_balancer(self, el):
        ip = None
        port = None
        sourceAddresses = {}
        if 'port' in el:
            port = el['port']
        if 'sourceAddresses' in el:
            sourceAddresses = el['sourceAddresses']
        extra = {'ipv6PublicSource': sourceAddresses.get('ipv6Public'), 'ipv4PublicSource': sourceAddresses.get('ipv4Public'), 'ipv4PrivateSource': sourceAddresses.get('ipv4Servicenet'), 'service_name': self.connection.get_service_name(), 'uri': 'https://%s%s/loadbalancers/%s' % (self.connection.host, self.connection.request_path, el['id'])}
        if 'virtualIps' in el:
            ip = el['virtualIps'][0]['address']
            extra['virtualIps'] = el['virtualIps']
        if 'protocol' in el:
            extra['protocol'] = el['protocol']
        if 'algorithm' in el and el['algorithm'] in self._VALUE_TO_ALGORITHM_MAP:
            extra['algorithm'] = self._value_to_algorithm(el['algorithm'])
        if 'healthMonitor' in el:
            health_monitor = self._to_health_monitor(el)
            if health_monitor:
                extra['healthMonitor'] = health_monitor
        if 'connectionThrottle' in el:
            extra['connectionThrottle'] = self._to_connection_throttle(el)
        if 'sessionPersistence' in el:
            persistence = el['sessionPersistence']
            extra['sessionPersistenceType'] = persistence.get('persistenceType')
        if 'connectionLogging' in el:
            logging = el['connectionLogging']
            extra['connectionLoggingEnabled'] = logging.get('enabled')
        if 'nodes' in el:
            extra['members'] = self._to_members(el)
        if 'created' in el:
            extra['created'] = self._iso_to_datetime(el['created']['time'])
        if 'updated' in el:
            extra['updated'] = self._iso_to_datetime(el['updated']['time'])
        if 'accessList' in el:
            extra['accessList'] = [self._to_access_rule(rule) for rule in el['accessList']]
        return LoadBalancer(id=el['id'], name=el['name'], state=self.LB_STATE_MAP.get(el['status'], State.UNKNOWN), ip=ip, port=port, driver=self.connection.driver, extra=extra)

    def _to_members(self, object, balancer=None):
        return [self._to_member(el, balancer) for el in object['nodes']]

    def _to_member(self, el, balancer=None):
        extra = {}
        if 'weight' in el:
            extra['weight'] = el['weight']
        if 'condition' in el and el['condition'] in self.LB_MEMBER_CONDITION_MAP:
            extra['condition'] = self.LB_MEMBER_CONDITION_MAP.get(el['condition'])
        if 'status' in el:
            extra['status'] = el['status']
        lbmember = Member(id=el['id'], ip=el['address'], port=el['port'], balancer=balancer, extra=extra)
        return lbmember

    def _protocol_to_value(self, protocol):
        non_standard_protocols = {'imapv2': 'IMAPv2', 'imapv3': 'IMAPv3', 'imapv4': 'IMAPv4'}
        protocol_name = protocol.lower()
        if protocol_name in non_standard_protocols:
            protocol_value = non_standard_protocols[protocol_name]
        else:
            protocol_value = protocol.upper()
        return protocol_value

    def _kwargs_to_mutable_attrs(self, **attrs):
        update_attrs = {}
        if 'name' in attrs:
            update_attrs['name'] = attrs['name']
        if 'algorithm' in attrs:
            algorithm_value = self._algorithm_to_value(attrs['algorithm'])
            update_attrs['algorithm'] = algorithm_value
        if 'protocol' in attrs:
            update_attrs['protocol'] = self._protocol_to_value(attrs['protocol'])
        if 'port' in attrs:
            update_attrs['port'] = int(attrs['port'])
        if 'vip' in attrs:
            if attrs['vip'] == 'PUBLIC' or attrs['vip'] == 'SERVICENET':
                update_attrs['virtualIps'] = [{'type': attrs['vip']}]
            else:
                update_attrs['virtualIps'] = [{'id': attrs['vip']}]
        return update_attrs

    def _kwargs_to_mutable_member_attrs(self, **attrs):
        update_attrs = {}
        if 'condition' in attrs:
            update_attrs['condition'] = self.CONDITION_LB_MEMBER_MAP.get(attrs['condition'])
        if 'weight' in attrs:
            update_attrs['weight'] = attrs['weight']
        return update_attrs

    def _to_health_monitor(self, el):
        health_monitor_data = el['healthMonitor']
        type = health_monitor_data.get('type')
        delay = health_monitor_data.get('delay')
        timeout = health_monitor_data.get('timeout')
        attempts_before_deactivation = health_monitor_data.get('attemptsBeforeDeactivation')
        if type == 'CONNECT':
            return RackspaceHealthMonitor(type=type, delay=delay, timeout=timeout, attempts_before_deactivation=attempts_before_deactivation)
        if type == 'HTTP' or type == 'HTTPS':
            return RackspaceHTTPHealthMonitor(type=type, delay=delay, timeout=timeout, attempts_before_deactivation=attempts_before_deactivation, path=health_monitor_data.get('path'), status_regex=health_monitor_data.get('statusRegex'), body_regex=health_monitor_data.get('bodyRegex', ''))
        return None

    def _to_connection_throttle(self, el):
        connection_throttle_data = el['connectionThrottle']
        min_connections = connection_throttle_data.get('minConnections')
        max_connections = connection_throttle_data.get('maxConnections')
        max_connection_rate = connection_throttle_data.get('maxConnectionRate')
        rate_interval = connection_throttle_data.get('rateInterval')
        return RackspaceConnectionThrottle(min_connections=min_connections, max_connections=max_connections, max_connection_rate=max_connection_rate, rate_interval_seconds=rate_interval)

    def _to_access_rule(self, el):
        return RackspaceAccessRule(id=el.get('id'), rule_type=self._to_access_rule_type(el.get('type')), address=el.get('address'))

    def _to_access_rule_type(self, type):
        if type == 'ALLOW':
            return RackspaceAccessRuleType.ALLOW
        elif type == 'DENY':
            return RackspaceAccessRuleType.DENY

    def _iso_to_datetime(self, isodate):
        date_formats = ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z')
        date = None
        for date_format in date_formats:
            try:
                date = datetime.strptime(isodate, date_format)
            except ValueError:
                pass
            if date:
                break
        return date
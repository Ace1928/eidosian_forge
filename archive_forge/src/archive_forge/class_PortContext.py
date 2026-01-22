import abc
from neutron_lib.api.definitions import portbindings
class PortContext(object, metaclass=abc.ABCMeta):
    """Context passed to MechanismDrivers for changes to port resources.

    A PortContext instance wraps a port resource. It provides helper
    methods for accessing other relevant information. Results from
    expensive operations are cached so that other MechanismDrivers can
    freely access the same information.
    """

    @property
    @abc.abstractmethod
    def current(self):
        """Return the port in its current configuration.

        Return the port, as defined by NeutronPluginBaseV2.
        create_port and all extensions in the ml2 plugin, with
        all its properties 'current' at the time the context was
        established.
        """

    @property
    @abc.abstractmethod
    def original(self):
        """Return the port in its original configuration.

        Return the port, with all its properties set to their
        original values prior to a call to update_port. Method is
        only valid within calls to update_port_precommit and
        update_port_postcommit.
        """

    @property
    @abc.abstractmethod
    def status(self):
        """Return the status of the current port."""

    @property
    @abc.abstractmethod
    def original_status(self):
        """Return the status of the original port.

        The method is only valid within calls to update_port_precommit and
        update_port_postcommit.
        """

    @property
    @abc.abstractmethod
    def network(self):
        """Return the NetworkContext associated with this port."""

    @property
    @abc.abstractmethod
    def binding_levels(self):
        """Return dictionaries describing the current binding levels.

        This property returns a list of dictionaries describing each
        binding level if the port is bound or partially bound, or None
        if the port is unbound. Each returned dictionary contains the
        name of the bound driver under the BOUND_DRIVER key, and the
        bound segment dictionary under the BOUND_SEGMENT key.

        The first entry (index 0) describes the top-level binding,
        which always involves one of the port's network's static
        segments. In the case of a hierarchical binding, subsequent
        entries describe the lower-level bindings in descending order,
        which may involve dynamic segments. Adjacent levels where
        different drivers bind the same static or dynamic segment are
        possible. The last entry (index -1) describes the bottom-level
        binding that supplied the port's binding:vif_type and
        binding:vif_details attribute values.

        Within calls to MechanismDriver.bind_port, descriptions of the
        levels above the level currently being bound are returned.
        """

    @property
    @abc.abstractmethod
    def original_binding_levels(self):
        """Return dictionaries describing the original binding levels.

        This property returns a list of dictionaries describing each
        original binding level if the port was previously bound, or
        None if the port was unbound. The content is as described for
        the binding_levels property.

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def top_bound_segment(self):
        """Return the current top-level bound segment dictionary.

        This property returns the current top-level bound segment
        dictionary, or None if the port is unbound. For a bound port,
        top_bound_segment is equivalent to
        binding_levels[0][BOUND_SEGMENT], and returns one of the
        port's network's static segments.
        """

    @property
    @abc.abstractmethod
    def original_top_bound_segment(self):
        """Return the original top-level bound segment dictionary.

        This property returns the original top-level bound segment
        dictionary, or None if the port was previously unbound. For a
        previously bound port, original_top_bound_segment is
        equivalent to original_binding_levels[0][BOUND_SEGMENT], and
        returns one of the port's network's static segments.

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def bottom_bound_segment(self):
        """Return the current bottom-level bound segment dictionary.

        This property returns the current bottom-level bound segment
        dictionary, or None if the port is unbound. For a bound port,
        bottom_bound_segment is equivalent to
        binding_levels[-1][BOUND_SEGMENT], and returns the segment
        whose binding supplied the port's binding:vif_type and
        binding:vif_details attribute values.
        """

    @property
    @abc.abstractmethod
    def original_bottom_bound_segment(self):
        """Return the original bottom-level bound segment dictionary.

        This property returns the original bottom-level bound segment
        dictionary, or None if the port was previously unbound. For a
        previously bound port, original_bottom_bound_segment is
        equivalent to original_binding_levels[-1][BOUND_SEGMENT], and
        returns the segment whose binding supplied the port's previous
        binding:vif_type and binding:vif_details attribute values.

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def host(self):
        """Return the host with which the port is associated.

        In the context of a host-specific operation on a distributed
        port, the host property indicates the host for which the port
        operation is being performed. Otherwise, it is the same value
        as current['binding:host_id'].
        """

    @property
    @abc.abstractmethod
    def original_host(self):
        """Return the original host with which the port was associated.

        In the context of a host-specific operation on a distributed
        port, the original_host property indicates the host for which
        the port operation is being performed. Otherwise, it is the
        same value as original['binding:host_id'].

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def vif_type(self):
        """Return the vif_type indicating the binding state of the port.

        In the context of a host-specific operation on a distributed
        port, the vif_type property indicates the binding state for
        the host for which the port operation is being
        performed. Otherwise, it is the same value as
        current['binding:vif_type'].
        """

    @property
    @abc.abstractmethod
    def original_vif_type(self):
        """Return the original vif_type of the port.

        In the context of a host-specific operation on a distributed
        port, the original_vif_type property indicates original
        binding state for the host for which the port operation is
        being performed. Otherwise, it is the same value as
        original['binding:vif_type'].

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def vif_details(self):
        """Return the vif_details describing the binding of the port.

        In the context of a host-specific operation on a distributed
        port, the vif_details property describes the binding for the
        host for which the port operation is being
        performed. Otherwise, it is the same value as
        current['binding:vif_details'].
        """

    @property
    @abc.abstractmethod
    def original_vif_details(self):
        """Return the original vif_details of the port.

        In the context of a host-specific operation on a distributed
        port, the original_vif_details property describes the original
        binding for the host for which the port operation is being
        performed. Otherwise, it is the same value as
        original['binding:vif_details'].

        This property is only valid within calls to
        update_port_precommit and update_port_postcommit. It returns
        None otherwise.
        """

    @property
    @abc.abstractmethod
    def segments_to_bind(self):
        """Return the list of segments with which to bind the port.

        This property returns the list of segment dictionaries with
        which the mechanism driver may bind the port. When
        establishing a top-level binding, these will be the port's
        network's static segments. For each subsequent level, these
        will be the segments passed to continue_binding by the
        mechanism driver that bound the level above.

        This property is only valid within calls to
        MechanismDriver.bind_port. It returns None otherwise.
        """

    @abc.abstractmethod
    def host_agents(self, agent_type):
        """Get agents of the specified type on port's host.

        :param agent_type: Agent type identifier
        :returns: List of neutron.db.models.agent.Agent records
        """

    @abc.abstractmethod
    def set_binding(self, segment_id, vif_type, vif_details, status=None):
        """Set the bottom-level binding for the port.

        :param segment_id: Network segment bound for the port.
        :param vif_type: The VIF type for the bound port.
        :param vif_details: Dictionary with details for VIF driver.
        :param status: Port status to set if not None.

        This method is called by MechanismDriver.bind_port to indicate
        success and specify binding details to use for port. The
        segment_id must identify an item in the current value of the
        segments_to_bind property.
        """

    @abc.abstractmethod
    def continue_binding(self, segment_id, next_segments_to_bind):
        """Continue binding the port with different segments.

        :param segment_id: Network segment partially bound for the port.
        :param next_segments_to_bind: Segments to continue binding with.

        This method is called by MechanismDriver.bind_port to indicate
        it was able to partially bind the port, but that one or more
        additional mechanism drivers are required to complete the
        binding. The segment_id must identify an item in the current
        value of the segments_to_bind property. The list of segments
        IDs passed as next_segments_to_bind identify dynamic (or
        static) segments of the port's network that will be used to
        populate segments_to_bind for the next lower level of a
        hierarchical binding.
        """

    @abc.abstractmethod
    def allocate_dynamic_segment(self, segment):
        """Allocate a dynamic segment.

        :param segment: A partially or fully specified segment dictionary

        Called by the MechanismDriver.bind_port, create_port or update_port
        to dynamically allocate a segment for the port using the partial
        segment specified. The segment dictionary can be a fully or partially
        specified segment. At a minimum it needs the network_type populated to
        call on the appropriate type driver.
        """

    @abc.abstractmethod
    def release_dynamic_segment(self, segment_id):
        """Release an allocated dynamic segment.

        :param segment_id: UUID of the dynamic network segment.

        Called by the MechanismDriver.delete_port or update_port to release
        the dynamic segment allocated for this port.
        """
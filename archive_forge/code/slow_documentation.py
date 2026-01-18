import struct
from . import packet_base
from os_ken.lib import addrconv
Link Aggregation Control Protocol(LACP, IEEE 802.1AX)
    header encoder/decoder class.

    http://standards.ieee.org/getieee802/download/802.1AX-2008.pdf

    LACPDU format

    +------------------------------------------------+--------+
    | LACPDU structure                               | Octets |
    +================================================+========+
    | Subtype = LACP                                 | 1      |
    +------------------------------------------------+--------+
    | Version Number                                 | 1      |
    +------------+-----------------------------------+--------+
    | TLV        | TLV_type = Actor Information      | 1      |
    | Actor      |                                   |        |
    +------------+-----------------------------------+--------+
    |            | Actor_Information_Length = 20     | 1      |
    +------------+-----------------------------------+--------+
    |            | Actor_System_Priority             | 2      |
    +------------+-----------------------------------+--------+
    |            | Actor_System                      | 6      |
    +------------+-----------------------------------+--------+
    |            | Actor_Key                         | 2      |
    +------------+-----------------------------------+--------+
    |            | Actor_Port_Priority               | 2      |
    +------------+-----------------------------------+--------+
    |            | Actor_Port                        | 2      |
    +------------+-----------------------------------+--------+
    |            | Actor_State                       | 1      |
    +------------+-----------------------------------+--------+
    |            | Reserved                          | 3      |
    +------------+-----------------------------------+--------+
    | TLV        | TLV_type = Partner Information    | 1      |
    | Partner    |                                   |        |
    +------------+-----------------------------------+--------+
    |            | Partner_Information_Length = 20   | 1      |
    +------------+-----------------------------------+--------+
    |            | Partner_System_Priority           | 2      |
    +------------+-----------------------------------+--------+
    |            | Partner_System                    | 6      |
    +------------+-----------------------------------+--------+
    |            | Partner_Key                       | 2      |
    +------------+-----------------------------------+--------+
    |            | Partner_Port_Priority             | 2      |
    +------------+-----------------------------------+--------+
    |            | Partner_Port                      | 2      |
    +------------+-----------------------------------+--------+
    |            | Partner_State                     | 1      |
    +------------+-----------------------------------+--------+
    |            | Reserved                          | 3      |
    +------------+-----------------------------------+--------+
    | TLV        | TLV_type = Collector Information  | 1      |
    | Collector  |                                   |        |
    +------------+-----------------------------------+--------+
    |            | Collector_Information_Length = 16 | 1      |
    +------------+-----------------------------------+--------+
    |            | Collector_Max_Delay               | 2      |
    +------------+-----------------------------------+--------+
    |            | Reserved                          | 12     |
    +------------+-----------------------------------+--------+
    | TLV        | TLV_type = Terminator             | 1      |
    | Terminator |                                   |        |
    +------------+-----------------------------------+--------+
    |            | Terminator_Length = 0             | 1      |
    +------------+-----------------------------------+--------+
    |            | Reserved                          | 50     |
    +------------+-----------------------------------+--------+


    Terminator information uses a length value of 0 (0x00).

    NOTE--The use of a Terminator_Length of 0 is intentional.
          In TLV encoding schemes it is common practice
          for the terminator encoding to be 0 both
          for the type and the length.

    Actor_State and Partner_State encoded as individual bits within
    a single octet as follows:

    +------+------+------+------+------+------+------+------+
    | 7    | 6    | 5    | 4    | 3    | 2    | 1    | 0    |
    +======+======+======+======+======+======+======+======+
    | EXPR | DFLT | DIST | CLCT | SYNC | AGGR | TMO  | ACT  |
    +------+------+------+------+------+------+------+------+

    ACT
        bit 0.
        about the activity control value with regard to this link.
    TMO
        bit 1.
        about the timeout control value with regard to this link.
    AGGR
        bit 2.
        about how the system regards this link from the point of view
        of the aggregation.
    SYNC
        bit 3.
        about how the system regards this link from the point of view
        of the synchronization.
    CLCT
        bit 4.
        about collecting of incoming frames.
    DIST
        bit 5.
        about distributing of outgoing frames.
    DFLT
        bit 6.
        about the opposite system information which the system use.
    EXPR
        bit 7.
        about the expire state of the system.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============================== ====================================
    Attribute                       Description
    =============================== ====================================
    version                         LACP version. This parameter must be
                                    set to LACP_VERSION_NUMBER(i.e. 1).

    actor_system_priority           The priority assigned to this
                                    System.

    actor_system                    The Actor's System ID, encoded as
                                    a MAC address.

    actor_key                       The operational Key value assigned
                                    to the port by the Actor.

    actor_port_priority             The priority assigned to this port.

    actor_port                      The port number assigned to the
                                    port by the Actor.

    actor_state_activity            .. _lacp_activity:

                                    about the activity control value
                                    with regard to this link.

                                    LACP_STATE_ACTIVE(1)

                                    LACP_STATE_PASSIVE(0)

    actor_state_timeout             .. _lacp_timeout:

                                    about the timeout control value
                                    with regard to this link.

                                    LACP_STATE_SHORT_TIMEOUT(1)

                                    LACP_STATE_LONG_TIMEOUT(0)

    actor_state_aggregation         .. _lacp_aggregation:

                                    about how the system regards this
                                    link from the point of view of the
                                    aggregation.

                                    LACP_STATE_AGGREGATEABLE(1)

                                    LACP_STATE_INDIVIDUAL(0)

    actor_state_synchronization     .. _lacp_synchronization:

                                    about how the system regards this
                                    link from the point of view of the
                                    synchronization.

                                    LACP_STATE_IN_SYNC(1)

                                    LACP_STATE_OUT_OF_SYNC(0)

    actor_state_collecting          .. _lacp_collecting:

                                    about collecting of incoming frames.

                                    LACP_STATE_COLLECTING_ENABLED(1)

                                    LACP_STATE_COLLECTING_DISABLED(0)

    actor_state_distributing        .. _lacp_distributing:

                                    about distributing of outgoing frames.

                                    LACP_STATE_DISTRIBUTING_ENABLED(1)

                                    LACP_STATE_DISTRIBUTING_DISABLED(0)

    actor_state_defaulted           .. _lacp_defaulted:

                                    about the Partner information
                                    which the the Actor use.

                                    LACP_STATE_DEFAULTED_PARTNER(1)

                                    LACP_STATE_OPERATIONAL_PARTNER(0)

    actor_state_expired             .. _lacp_expired:

                                    about the state of the Actor.

                                    LACP_STATE_EXPIRED(1)

                                    LACP_STATE_NOT_EXPIRED(0)

    partner_system_priority         The priority assigned to the
                                    Partner System.

    partner_system                  The Partner's System ID, encoded
                                    as a MAC address.

    partner_key                     The operational Key value assigned
                                    to the port by the Partner.

    partner_port_priority           The priority assigned to this port
                                    by the Partner.

    partner_port                    The port number assigned to the
                                    port by the Partner.

    partner_state_activity          See :ref:`actor_state_activity                                    <lacp_activity>`.

    partner_state_timeout           See :ref:`actor_state_timeout                                    <lacp_timeout>`.

    partner_state_aggregation       See :ref:`actor_state_aggregation                                    <lacp_aggregation>`.

    partner_state_synchronization   See
                                    :ref:`actor_state_synchronization                                    <lacp_synchronization>`.

    partner_state_collecting        See :ref:`actor_state_collecting                                    <lacp_collecting>`.

    partner_state_distributing      See :ref:`actor_state_distributing                                    <lacp_distributing>`.

    partner_state_defaulted         See :ref:`actor_state_defaulted                                    <lacp_defaulted>`.

    partner_state_expired           See :ref:`actor_state_expired                                    <lacp_expired>`.

    collector_max_delay             the maximum time that the Frame
                                    Collector may delay.
    =============================== ====================================

    
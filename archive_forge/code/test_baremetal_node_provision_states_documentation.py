from ironicclient.tests.functional.osc.v1 import base
Deploy, rebuild and undeploy node.

        Test steps:
        1) Create baremetal node in setUp.
        2) Check initial "enroll" provision state.
        3) Set baremetal node "manage" provision state.
        4) Check baremetal node provision_state field value is "manageable".
        5) Set baremetal node "provide" provision state.
        6) Check baremetal node provision_state field value is "available".
        7) Set baremetal node "deploy" provision state.
        8) Check baremetal node provision_state field value is "active".
        9) Set baremetal node "rebuild" provision state.
        10) Check baremetal node provision_state field value is "active".
        11) Set baremetal node "undeploy" provision state.
        12) Check baremetal node provision_state field value is "available".
        13) Set baremetal node "manage" provision state.
        14) Check baremetal node provision_state field value is "manageable".
        15) Set baremetal node "provide" provision state.
        16) Check baremetal node provision_state field value is "available".
        
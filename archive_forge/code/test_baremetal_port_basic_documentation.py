from ironicclient.tests.functional.osc.v1 import base
Create port with specific port group UUID.

        Test steps:
        1) Create node in setUp().
        2) Create a port group.
        3) Create a port with specified port group.
        4) Check port properties for portgroup_uuid.
        
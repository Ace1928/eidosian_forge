from cinderclient.tests.functional import base
Basic read-only test for cinderclient.

    Simple check of base list commands, verify they
    respond and include the expected headers in the
    resultant table.

    Not intended for testing things that require actual
    resource creation/manipulation, thus the name 'read-only'.

    
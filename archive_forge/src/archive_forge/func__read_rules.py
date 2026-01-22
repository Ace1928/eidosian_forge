import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _read_rules(self, path):
    """Read and parse rules from path

        Expect the file to contain a valid JSON structure.

        :param path: path to the file
        :return: loaded and valid dictionary with rules
        :raises exception.CommandError: In case the file cannot be
            accessed or the content is not a valid JSON.

        Example of the content of the file:
            [
                {
                    "local": [
                        {
                            "group": {
                                "id": "85a868"
                            }
                        }
                    ],
                    "remote": [
                        {
                            "type": "orgPersonType",
                            "any_one_of": [
                                "Employee"
                            ]
                        },
                        {
                            "type": "sn",
                            "any_one_of": [
                                "Young"
                            ]
                        }
                    ]
                }
            ]

        """
    blob = utils.read_blob_file_contents(path)
    try:
        rules = json.loads(blob)
    except ValueError as e:
        msg = _('An error occurred when reading rules from file %(path)s: %(error)s') % {'path': path, 'error': e}
        raise exceptions.CommandError(msg)
    else:
        return rules
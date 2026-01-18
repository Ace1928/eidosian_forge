from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.common.text.converters import to_native
Manages Rundeck API requests via HTTP(S)

    :arg module:   The AnsibleModule (used to get url, api_version, api_token, etc).
    :arg endpoint: The API endpoint to be used.
    :kwarg data:   The data to be sent (in case of POST/PUT).
    :kwarg method: "POST", "PUT", etc.

    :returns: A tuple of (**response**, **info**). Use ``response.read()`` to read the data.
        The **info** contains the 'status' and other meta data. When a HttpError (status >= 400)
        occurred then ``info['body']`` contains the error response data::

    Example::

        data={...}
        resp, info = fetch_url(module,
                               "http://rundeck.example.org",
                               data=module.jsonify(data),
                               method="POST")
        status_code = info["status"]
        body = resp.read()
        if status_code >= 400 :
            body = info['body']
    
from dash.development.base_component import Component, _explicitize_args
A ConfirmDialog component.
    ConfirmDialog is used to display the browser's native "confirm" modal,
    with an optional message and two buttons ("OK" and "Cancel").
    This ConfirmDialog can be used in conjunction with buttons when the user
    is performing an action that should require an extra step of verification.

    Keyword arguments:

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - cancel_n_clicks (number; default 0):
        Number of times the popup was canceled.

    - cancel_n_clicks_timestamp (number; default -1):
        Last time the cancel button was clicked.

    - displayed (boolean; optional):
        Set to True to send the ConfirmDialog.

    - message (string; optional):
        Message to show in the popup.

    - submit_n_clicks (number; default 0):
        Number of times the submit button was clicked.

    - submit_n_clicks_timestamp (number; default -1):
        Last time the submit button was clicked.
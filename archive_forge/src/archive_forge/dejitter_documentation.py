from kivy.config import Config

    Get rid of jitterish BLOBs.
    Example::

        [postproc]
        jitter_distance = 0.004
        jitter_ignore_devices = mouse,mactouch

    :Configuration:
        `jitter_distance`: float
            A float in range 0-1.
        `jitter_ignore_devices`: string
            A comma-separated list of device identifiers that
            should not be processed by dejitter (because they're
            very precise already).
    
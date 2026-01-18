import abc
Starts ingesting data.

        This may start a background thread or process, and will return
        once communication with that task is established. It won't block
        forever as data is reloaded.

        Must only be called once.
        
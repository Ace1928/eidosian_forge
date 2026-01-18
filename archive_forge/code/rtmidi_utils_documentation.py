Expand ALSA port name.

    RtMidi/ALSA includes client name and client:port number in
    the port name, for example:

        TiMidity:TiMidity port 0 128:0

    This allows you to specify only port name or client:port name when
    opening a port. It will compare the name to each name in
    port_names (typically returned from get_*_names()) and try these
    three variants in turn:

        TiMidity:TiMidity port 0 128:0
        TiMidity:TiMidity port 0
        TiMidity port 0

    It returns the first match. If no match is found it returns the
    passed name so the caller can deal with it.
    
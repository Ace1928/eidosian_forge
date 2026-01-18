from twisted.python import log

        Send a message event to the I{syslog}.

        @param eventDict: The event to send.  If it has no C{'message'} key, it
            will be ignored.  Otherwise, if it has C{'syslogPriority'} and/or
            C{'syslogFacility'} keys, these will be used as the syslog priority
            and facility.  If it has no C{'syslogPriority'} key but a true
            value for the C{'isError'} key, the B{LOG_ALERT} priority will be
            used; if it has a false value for C{'isError'}, B{LOG_INFO} will be
            used.  If the C{'message'} key is multiline, each line will be sent
            to the syslog separately.
        
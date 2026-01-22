from constantly import NamedConstant, Names

        Get the log level with the given name.

        @param name: The name of a log level.

        @return: The L{LogLevel} with the specified C{name}.

        @raise InvalidLogLevelError: if the C{name} does not name a valid log
            level.
        
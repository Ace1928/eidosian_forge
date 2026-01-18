import abc

        Measure this quantity and return the result

        Arguments:
            config (MetricConfig): The configuration for this metric
            now (int): The POSIX time in milliseconds the measurement
                is being taken

        Returns:
            The measured value
        
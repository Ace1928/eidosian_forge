class _ArgParser:
    """Internal argument parser implementation function object."""

    def __init__(self, method_name, param_defs, external_param_processor):
        self.__method_name = method_name
        self.__param_defs = param_defs
        self.__external_param_processor = external_param_processor
        self.__stack = []

    def __call__(self, args, kwargs, extra_parameter_errors):
        """
        Runs the main argument parsing operation.

        Passed args & kwargs objects are not modified during parsing.

        Returns an informative 2-tuple containing the number of required &
        allowed arguments.

        """
        assert not self.active(), 'recursive argument parsing not allowed'
        self.__init_run(args, kwargs, extra_parameter_errors)
        try:
            self.__process_parameters()
            return self.__all_parameters_processed()
        finally:
            self.__cleanup_run()
            assert not self.active()

    def active(self):
        """
        Return whether this object is currently running argument processing.

        Used to avoid recursively entering argument processing from within an
        external parameter processor.

        """
        return bool(self.__stack)

    def __all_parameters_processed(self):
        """
        Finish the argument processing.

        Should be called after all the web service operation's parameters have
        been successfully processed and, afterwards, no further parameter
        processing is allowed.

        Returns a 2-tuple containing the number of required & allowed
        arguments.

        See the _ArgParser class description for more detailed information.

        """
        assert self.active()
        sentinel_frame = self.__stack[0]
        self.__pop_frames_above(sentinel_frame)
        assert len(self.__stack) == 1
        self.__pop_top_frame()
        assert not self.active()
        args_required = sentinel_frame.args_required()
        args_allowed = sentinel_frame.args_allowed()
        self.__check_for_extra_arguments(args_required, args_allowed)
        return (args_required, args_allowed)

    def __check_for_extra_arguments(self, args_required, args_allowed):
        """
        Report an error in case any extra arguments are detected.

        Does nothing if reporting extra arguments as exceptions has not been
        enabled.

        May only be called after the argument processing has been completed.

        """
        assert not self.active()
        if not self.__extra_parameter_errors:
            return
        if self.__kwargs:
            param_name = list(self.__kwargs.keys())[0]
            if param_name in self.__params_with_arguments:
                msg = "got multiple values for parameter '%s'"
            else:
                msg = "got an unexpected keyword argument '%s'"
            self.__error(msg % (param_name,))
        if self.__args:

            def plural_suffix(count):
                if count == 1:
                    return ''
                return 's'

            def plural_was_were(count):
                if count == 1:
                    return 'was'
                return 'were'
            expected = args_required
            if args_required != args_allowed:
                expected = '%d to %d' % (args_required, args_allowed)
            given = self.__args_count
            msg_parts = ['takes %s positional argument' % (expected,), plural_suffix(expected), ' but %d ' % (given,), plural_was_were(given), ' given']
            self.__error(''.join(msg_parts))

    def __cleanup_run(self):
        """Cleans up after a completed argument parsing run."""
        self.__stack = []
        assert not self.active()

    def __error(self, message):
        """Report an argument processing error."""
        raise TypeError('%s() %s' % (self.__method_name, message))

    def __frame_factory(self, ancestry_item):
        """Construct a new frame representing the given ancestry item."""
        frame_class = Frame
        if ancestry_item is not None and ancestry_item.choice():
            frame_class = ChoiceFrame
        return frame_class(ancestry_item, self.__error, self.__extra_parameter_errors)

    def __get_param_value(self, name):
        """
        Extract a parameter value from the remaining given arguments.

        Returns a 2-tuple consisting of the following:
          * Boolean indicating whether an argument has been specified for the
            requested input parameter.
          * Parameter value.

        """
        if self.__args:
            return (True, self.__args.pop(0))
        try:
            value = self.__kwargs.pop(name)
        except KeyError:
            return (False, None)
        return (True, value)

    def __in_choice_context(self):
        """
        Whether we are currently processing a choice parameter group.

        This includes processing a parameter defined directly or indirectly
        within such a group.

        May only be called during parameter processing or the result will be
        calculated based on the context left behind by the previous parameter
        processing if any.

        """
        for x in self.__stack:
            if x.__class__ is ChoiceFrame:
                return True
        return False

    def __init_run(self, args, kwargs, extra_parameter_errors):
        """Initializes data for a new argument parsing run."""
        assert not self.active()
        self.__args = list(args)
        self.__kwargs = dict(kwargs)
        self.__extra_parameter_errors = extra_parameter_errors
        self.__args_count = len(args) + len(kwargs)
        self.__params_with_arguments = set()
        self.__stack = []
        self.__push_frame(None)

    def __match_ancestry(self, ancestry):
        """
        Find frames matching the given ancestry.

        Returns a tuple containing the following:
          * Topmost frame matching the given ancestry or the bottom-most sentry
            frame if no frame matches.
          * Unmatched ancestry part.

        """
        stack = self.__stack
        if len(stack) == 1:
            return (stack[0], ancestry)
        previous = stack[0]
        for frame, n in zip(stack[1:], range(len(ancestry))):
            if frame.id() is not ancestry[n]:
                return (previous, ancestry[n:])
            previous = frame
        return (frame, ancestry[n + 1:])

    def __pop_frames_above(self, frame):
        """Pops all the frames above, but not including the given frame."""
        while self.__stack[-1] is not frame:
            self.__pop_top_frame()
        assert self.__stack

    def __pop_top_frame(self):
        """Pops the top frame off the frame stack."""
        popped = self.__stack.pop()
        if self.__stack:
            self.__stack[-1].process_subframe(popped)

    def __process_parameter(self, param_name, param_type, ancestry=None):
        """Collect values for a given web service operation input parameter."""
        assert self.active()
        param_optional = param_type.optional()
        has_argument, value = self.__get_param_value(param_name)
        if has_argument:
            self.__params_with_arguments.add(param_name)
        self.__update_context(ancestry)
        self.__stack[-1].process_parameter(param_optional, value is not None)
        self.__external_param_processor(param_name, param_type, self.__in_choice_context(), value)

    def __process_parameters(self):
        """Collect values for given web service operation input parameters."""
        for pdef in self.__param_defs:
            self.__process_parameter(*pdef)

    def __push_frame(self, ancestry_item):
        """Push a new frame on top of the frame stack."""
        frame = self.__frame_factory(ancestry_item)
        self.__stack.append(frame)

    def __push_frames(self, ancestry):
        """
        Push new frames representing given ancestry items.

        May only be given ancestry items other than None. Ancestry item None
        represents the internal sentinel item and should never appear in a
        given parameter's ancestry information.

        """
        for x in ancestry:
            assert x is not None
            self.__push_frame(x)

    def __update_context(self, ancestry):
        if not ancestry:
            return
        match_result = self.__match_ancestry(ancestry)
        last_matching_frame, unmatched_ancestry = match_result
        self.__pop_frames_above(last_matching_frame)
        self.__push_frames(unmatched_ancestry)
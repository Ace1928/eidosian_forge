import warnings
class ConfirmationUserInterfacePolicy:
    """Wrapper for a UIFactory that allows or denies all confirmed actions."""

    def __init__(self, wrapped_ui, default_answer, specific_answers):
        """Generate a proxy UI that does no confirmations.

        Args:
          wrapped_ui: Underlying UIFactory.
          default_answer: Bool for whether requests for
            confirmation from the user should be noninteractively accepted or
            denied.
          specific_answers: Map from confirmation_id to bool answer.
        """
        self.wrapped_ui = wrapped_ui
        self.default_answer = default_answer
        self.specific_answers = specific_answers

    def __getattr__(self, name):
        return getattr(self.wrapped_ui, name)

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__, self.wrapped_ui, self.default_answer, self.specific_answers)

    def confirm_action(self, prompt, confirmation_id, prompt_kwargs):
        if confirmation_id in self.specific_answers:
            return self.specific_answers[confirmation_id]
        elif self.default_answer is not None:
            return self.default_answer
        else:
            return self.wrapped_ui.confirm_action(prompt, confirmation_id, prompt_kwargs)
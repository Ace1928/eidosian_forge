import statemachine
class VendingMachine(VendingMachineStateMixin):

    def __init__(self):
        self.initialize_state(Idle)
        self._pressed = None
        self._alpha_pressed = None
        self._digit_pressed = None

    def press_button(self, button):
        if button in 'ABCD':
            self._pressed = button
            self.press_alpha_button()
        elif button in '1234':
            self._pressed = button
            self.press_digit_button()
        else:
            print('Did not recognize button {!r}'.format(str(button)))

    def press_alpha_button(self):
        try:
            super(VendingMachine, self).press_alpha_button()
        except VendingMachineState.InvalidTransitionException as ite:
            print(ite)
        else:
            self._alpha_pressed = self._pressed

    def press_digit_button(self):
        try:
            super(VendingMachine, self).press_digit_button()
        except VendingMachineState.InvalidTransitionException as ite:
            print(ite)
        else:
            self._digit_pressed = self._pressed
            self.dispense()

    def dispense(self):
        try:
            super(VendingMachine, self).dispense()
        except VendingMachineState.InvalidTransitionException as ite:
            print(ite)
        else:
            print('Dispensing at {}{}'.format(self._alpha_pressed, self._digit_pressed))
            self._alpha_pressed = self._digit_pressed = None
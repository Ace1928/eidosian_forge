import statemachine
import librarybookstate
class Book(librarybookstate.BookStateMixin):

    def __init__(self):
        self.initialize_state(librarybookstate.New)
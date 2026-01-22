import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrsDrawer:
    BUFFER = 3
    TOPSPACE = 10
    OUTERSPACE = 6

    def __init__(self, drs, size_canvas=True, canvas=None):
        """
        :param drs: ``DrtExpression``, The DRS to be drawn
        :param size_canvas: bool, True if the canvas size should be the exact size of the DRS
        :param canvas: ``Canvas`` The canvas on which to draw the DRS.  If none is given, create a new canvas.
        """
        master = None
        if not canvas:
            master = Tk()
            master.title('DRT')
            font = Font(family='helvetica', size=12)
            if size_canvas:
                canvas = Canvas(master, width=0, height=0)
                canvas.font = font
                self.canvas = canvas
                right, bottom = self._visit(drs, self.OUTERSPACE, self.TOPSPACE)
                width = max(right + self.OUTERSPACE, 100)
                height = bottom + self.OUTERSPACE
                canvas = Canvas(master, width=width, height=height)
            else:
                canvas = Canvas(master, width=300, height=300)
            canvas.pack()
            canvas.font = font
        self.canvas = canvas
        self.drs = drs
        self.master = master

    def _get_text_height(self):
        """Get the height of a line of text"""
        return self.canvas.font.metrics('linespace')

    def draw(self, x=OUTERSPACE, y=TOPSPACE):
        """Draw the DRS"""
        self._handle(self.drs, self._draw_command, x, y)
        if self.master and (not in_idle()):
            self.master.mainloop()
        else:
            return self._visit(self.drs, x, y)

    def _visit(self, expression, x, y):
        """
        Return the bottom-rightmost point without actually drawing the item

        :param expression: the item to visit
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        return self._handle(expression, self._visit_command, x, y)

    def _draw_command(self, item, x, y):
        """
        Draw the given item at the given location

        :param item: the item to draw
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if isinstance(item, str):
            self.canvas.create_text(x, y, anchor='nw', font=self.canvas.font, text=item)
        elif isinstance(item, tuple):
            right, bottom = item
            self.canvas.create_rectangle(x, y, right, bottom)
            horiz_line_y = y + self._get_text_height() + self.BUFFER * 2
            self.canvas.create_line(x, horiz_line_y, right, horiz_line_y)
        return self._visit_command(item, x, y)

    def _visit_command(self, item, x, y):
        """
        Return the bottom-rightmost point without actually drawing the item

        :param item: the item to visit
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if isinstance(item, str):
            return (x + self.canvas.font.measure(item), y + self._get_text_height())
        elif isinstance(item, tuple):
            return item

    def _handle(self, expression, command, x=0, y=0):
        """
        :param expression: the expression to handle
        :param command: the function to apply, either _draw_command or _visit_command
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if command == self._visit_command:
            try:
                right = expression._drawing_width + x
                bottom = expression._drawing_height + y
                return (right, bottom)
            except AttributeError:
                pass
        if isinstance(expression, DrtAbstractVariableExpression):
            factory = self._handle_VariableExpression
        elif isinstance(expression, DRS):
            factory = self._handle_DRS
        elif isinstance(expression, DrtNegatedExpression):
            factory = self._handle_NegatedExpression
        elif isinstance(expression, DrtLambdaExpression):
            factory = self._handle_LambdaExpression
        elif isinstance(expression, BinaryExpression):
            factory = self._handle_BinaryExpression
        elif isinstance(expression, DrtApplicationExpression):
            factory = self._handle_ApplicationExpression
        elif isinstance(expression, PossibleAntecedents):
            factory = self._handle_VariableExpression
        elif isinstance(expression, DrtProposition):
            factory = self._handle_DrtProposition
        else:
            raise Exception(expression.__class__.__name__)
        right, bottom = factory(expression, command, x, y)
        expression._drawing_width = right - x
        expression._drawing_height = bottom - y
        return (right, bottom)

    def _handle_VariableExpression(self, expression, command, x, y):
        return command('%s' % expression, x, y)

    def _handle_NegatedExpression(self, expression, command, x, y):
        right = self._visit_command(DrtTokens.NOT, x, y)[0]
        right, bottom = self._handle(expression.term, command, right, y)
        command(DrtTokens.NOT, x, self._get_centered_top(y, bottom - y, self._get_text_height()))
        return (right, bottom)

    def _handle_DRS(self, expression, command, x, y):
        left = x + self.BUFFER
        bottom = y + self.BUFFER
        if expression.refs:
            refs = ' '.join(('%s' % r for r in expression.refs))
        else:
            refs = '     '
        max_right, bottom = command(refs, left, bottom)
        bottom += self.BUFFER * 2
        if expression.conds:
            for cond in expression.conds:
                right, bottom = self._handle(cond, command, left, bottom)
                max_right = max(max_right, right)
                bottom += self.BUFFER
        else:
            bottom += self._get_text_height() + self.BUFFER
        max_right += self.BUFFER
        return command((max_right, bottom), x, y)

    def _handle_ApplicationExpression(self, expression, command, x, y):
        function, args = expression.uncurry()
        if not isinstance(function, DrtAbstractVariableExpression):
            function = expression.function
            args = [expression.argument]
        function_bottom = self._visit(function, x, y)[1]
        max_bottom = max([function_bottom] + [self._visit(arg, x, y)[1] for arg in args])
        line_height = max_bottom - y
        function_drawing_top = self._get_centered_top(y, line_height, function._drawing_height)
        right = self._handle(function, command, x, function_drawing_top)[0]
        centred_string_top = self._get_centered_top(y, line_height, self._get_text_height())
        right = command(DrtTokens.OPEN, right, centred_string_top)[0]
        for i, arg in enumerate(args):
            arg_drawing_top = self._get_centered_top(y, line_height, arg._drawing_height)
            right = self._handle(arg, command, right, arg_drawing_top)[0]
            if i + 1 < len(args):
                right = command(DrtTokens.COMMA + ' ', right, centred_string_top)[0]
        right = command(DrtTokens.CLOSE, right, centred_string_top)[0]
        return (right, max_bottom)

    def _handle_LambdaExpression(self, expression, command, x, y):
        variables = DrtTokens.LAMBDA + '%s' % expression.variable + DrtTokens.DOT
        right = self._visit_command(variables, x, y)[0]
        right, bottom = self._handle(expression.term, command, right, y)
        command(variables, x, self._get_centered_top(y, bottom - y, self._get_text_height()))
        return (right, bottom)

    def _handle_BinaryExpression(self, expression, command, x, y):
        first_height = self._visit(expression.first, 0, 0)[1]
        second_height = self._visit(expression.second, 0, 0)[1]
        line_height = max(first_height, second_height)
        centred_string_top = self._get_centered_top(y, line_height, self._get_text_height())
        right = command(DrtTokens.OPEN, x, centred_string_top)[0]
        first_height = expression.first._drawing_height
        right, first_bottom = self._handle(expression.first, command, right, self._get_centered_top(y, line_height, first_height))
        right = command(' %s ' % expression.getOp(), right, centred_string_top)[0]
        second_height = expression.second._drawing_height
        right, second_bottom = self._handle(expression.second, command, right, self._get_centered_top(y, line_height, second_height))
        right = command(DrtTokens.CLOSE, right, centred_string_top)[0]
        return (right, max(first_bottom, second_bottom))

    def _handle_DrtProposition(self, expression, command, x, y):
        right = command(expression.variable, x, y)[0]
        right, bottom = self._handle(expression.term, command, right, y)
        return (right, bottom)

    def _get_centered_top(self, top, full_height, item_height):
        """Get the y-coordinate of the point that a figure should start at if
        its height is 'item_height' and it needs to be centered in an area that
        starts at 'top' and is 'full_height' tall."""
        return top + (full_height - item_height) / 2
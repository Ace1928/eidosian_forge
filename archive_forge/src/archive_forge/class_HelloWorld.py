from boto.pyami.scriptbase import ScriptBase
class HelloWorld(ScriptBase):

    def main(self):
        self.log('Hello World!!!')
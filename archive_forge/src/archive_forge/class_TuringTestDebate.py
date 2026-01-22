import random
class TuringTestDebate:

    def __init__(self):
        self.topics = ['Should AI have rights similar to humans if they can pass the Turing Test?', 'What are the potential risks of AI in society if they achieve human-like intelligence?', 'How should AI be regulated?']
        self.current_topic = None

    def select_topic(self):
        print('Select a topic to debate:')
        for index, topic in enumerate(self.topics):
            print(f'{index + 1}. {topic}')
        choice = int(input('Enter your choice (1-3): '))
        self.current_topic = self.topics[choice - 1]
        print(f'Topic selected: {self.current_topic}')

    def run_debate(self):
        print(f'Discussing: {self.current_topic}')
        for _ in range(3):
            user_input = input('Enter your argument: ')
            print('Analyzing argument...')
            if 'rights' in user_input.lower():
                print('Interesting point on rights. How do you think this aligns with human rights?')
            else:
                print("Thank you for your input. Let's consider other aspects as well.")